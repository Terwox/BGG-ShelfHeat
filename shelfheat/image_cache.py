"""
BGG box art image cache.

Downloads and caches game box art thumbnails from BGG CDN URLs
for use in CLIP image-to-image matching. Images are stored locally
at ~/.cache/shelfheat/images/{bgg_id}.jpg.
"""

import time
from pathlib import Path

import requests
from PIL import Image

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "shelfheat" / "images"
_DOWNLOAD_DELAY = 0.1  # seconds between downloads (be polite to BGG CDN)
_TIMEOUT = 15
_TARGET_SIZE = 256  # store at 256x256 for CLIP flexibility


def ensure_collection_images(
    games: list[dict],
    cache_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Download and cache box art for each game in the collection.

    Args:
        games: List of game dicts, each with "name", "bgg_id", and
               optional "image" (thumbnail URL from BGG API/CSV).
        cache_dir: Override cache directory. Defaults to ~/.cache/shelfheat/images/.

    Returns:
        {game_name: local_image_path} for games with cached images.
    """
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, Path] = {}
    to_download: list[dict] = []

    for game in games:
        bgg_id = game.get("bgg_id")
        name = game.get("name")
        url = game.get("image") or game.get("thumbnail")

        if not bgg_id or not name or not url:
            continue

        local_path = cache / f"{bgg_id}.jpg"
        if local_path.exists() and local_path.stat().st_size > 0:
            result[name] = local_path
        else:
            to_download.append({"name": name, "bgg_id": bgg_id, "url": url, "path": local_path})

    if to_download:
        print(f"[image_cache] Downloading {len(to_download)} box art images...")
        downloaded = 0
        failed = 0

        for i, item in enumerate(to_download):
            ok = _download_image(item["url"], item["path"])
            if ok:
                result[item["name"]] = item["path"]
                downloaded += 1
            else:
                failed += 1

            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(to_download)}] downloaded={downloaded} failed={failed}")

            if i < len(to_download) - 1:
                time.sleep(_DOWNLOAD_DELAY)

        print(f"[image_cache] Done: {downloaded} downloaded, {failed} failed, {len(result)} total cached")
    else:
        print(f"[image_cache] All {len(result)} images already cached")

    return result


def _download_image(url: str, dest: Path) -> bool:
    """Download a single image, resize, and save as JPEG."""
    try:
        resp = requests.get(url, timeout=_TIMEOUT, stream=True)
        resp.raise_for_status()

        # Save raw bytes to a temp file first, then process
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate and resize
        img = Image.open(tmp).convert("RGB")
        # Resize preserving aspect ratio, fitting within TARGET_SIZE
        img.thumbnail((_TARGET_SIZE, _TARGET_SIZE), Image.LANCZOS)
        img.save(dest, "JPEG", quality=85)
        tmp.unlink(missing_ok=True)
        return True

    except Exception as e:
        # Clean up on failure
        dest.unlink(missing_ok=True)
        dest.with_suffix(".tmp").unlink(missing_ok=True)
        return False


def enrich_from_bggdb(games: list[dict]) -> list[dict]:
    """
    For games that have a bgg_id but no image URL, look up the thumbnail
    from the local BGG database (bggdb). Modifies games in-place and returns them.
    """
    missing = [g for g in games if g.get("bgg_id") and not g.get("image") and not g.get("thumbnail")]
    if not missing:
        return games

    try:
        from shelfheat.bggdb import BGGLocalDB
        db = BGGLocalDB()
        found = 0
        for game in missing:
            info = db.lookup_id(game["bgg_id"])
            if info and info.get("thumbnail"):
                game["image"] = info["thumbnail"]
                found += 1
        db.close()
        if found:
            print(f"[image_cache] Enriched {found}/{len(missing)} games with thumbnails from local DB")
    except Exception as e:
        print(f"[image_cache] Could not enrich from bggdb: {e}")

    return games


def get_cached_path(bgg_id: int, cache_dir: str | Path | None = None) -> Path | None:
    """Return the local path for a cached image, or None if not cached."""
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    path = cache / f"{bgg_id}.jpg"
    return path if path.exists() and path.stat().st_size > 0 else None
