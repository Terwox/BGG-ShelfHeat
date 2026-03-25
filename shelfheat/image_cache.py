"""
BGG box art image cache.

Downloads and caches game box art from BGG:
  1. Front cover thumbnails (from BGG API/CSV thumbnail URLs)
  2. Gallery images (3D covers, spine shots, alternate angles) via the
     undocumented BGG gallery API, filtered by caption keywords.

Images are stored at ~/.cache/shelfheat/images/:
  - {bgg_id}.jpg            — front cover thumbnail
  - {bgg_id}_gallery_{imageid}.jpg — gallery images (3D, spine, etc.)
"""

import glob as glob_mod
import time
from pathlib import Path

import requests
from PIL import Image

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "shelfheat" / "images"
_DOWNLOAD_DELAY = 0.1  # seconds between CDN image downloads
_GALLERY_API_DELAY = 0.5  # seconds between gallery API calls
_TIMEOUT = 15
_TARGET_SIZE = 256  # store at 256x256 for CLIP flexibility

# Gallery API config
_GALLERY_API = "https://api.geekdo.com/api/images"
_GALLERY_PAGE_SIZE = 50
_GALLERY_MAX_PAGES = 10  # 10 × 50 = 500 images scanned per game
_GALLERY_MAX_DOWNLOADS = 5  # max gallery images to keep per game
_CAPTION_KEYWORDS = ["3d", "cover", "box", "spine", "side"]


def ensure_collection_images(
    games: list[dict],
    cache_dir: str | Path | None = None,
    include_gallery: bool = True,
) -> dict[str, list[Path]]:
    """
    Download and cache box art for each game in the collection.

    Args:
        games: List of game dicts, each with "name", "bgg_id", and
               optional "image" (thumbnail URL from BGG API/CSV).
        cache_dir: Override cache directory.
        include_gallery: If True, also fetch gallery images (3D covers, etc.)

    Returns:
        {game_name: [list of local image paths]} for games with cached images.
    """
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    # Phase 1: Download front cover thumbnails
    result: dict[str, list[Path]] = {}
    to_download: list[dict] = []

    for game in games:
        bgg_id = game.get("bgg_id")
        name = game.get("name")
        url = game.get("image") or game.get("thumbnail")

        if not bgg_id or not name:
            continue

        local_path = cache / f"{bgg_id}.jpg"
        paths = []

        if local_path.exists() and local_path.stat().st_size > 0:
            paths.append(local_path)
        elif url:
            to_download.append({"name": name, "bgg_id": bgg_id, "url": url, "path": local_path})

        # Also collect any existing gallery images
        gallery_pattern = str(cache / f"{bgg_id}_gallery_*.jpg")
        gallery_files = sorted(glob_mod.glob(gallery_pattern))
        paths.extend(Path(f) for f in gallery_files)

        if paths:
            result[name] = paths

    if to_download:
        print(f"[image_cache] Downloading {len(to_download)} front cover images...")
        downloaded = 0
        failed = 0

        for i, item in enumerate(to_download):
            ok = _download_image(item["url"], item["path"])
            if ok:
                result.setdefault(item["name"], []).insert(0, item["path"])
                downloaded += 1
            else:
                failed += 1

            if (i + 1) % 25 == 0:
                print(f"  [{i + 1}/{len(to_download)}] downloaded={downloaded} failed={failed}")

            if i < len(to_download) - 1:
                time.sleep(_DOWNLOAD_DELAY)

        print(f"[image_cache] Covers: {downloaded} downloaded, {failed} failed")
    else:
        total_covers = sum(1 for paths in result.values() if any(not "_gallery_" in str(p) for p in paths))
        print(f"[image_cache] All {total_covers} front covers already cached")

    # Phase 2: Fetch gallery images (3D covers, spine shots, etc.)
    if include_gallery:
        _fetch_all_gallery_images(games, cache, result)

    total_images = sum(len(paths) for paths in result.values())
    total_games = len(result)
    print(f"[image_cache] Total: {total_images} images for {total_games} games")

    return result


def _fetch_all_gallery_images(
    games: list[dict],
    cache: Path,
    result: dict[str, list[Path]],
):
    """Fetch gallery images for all games that don't already have them cached."""
    to_fetch = []
    for game in games:
        bgg_id = game.get("bgg_id")
        name = game.get("name")
        if not bgg_id or not name:
            continue

        # Skip if we already have gallery images cached
        gallery_pattern = str(cache / f"{bgg_id}_gallery_*.jpg")
        existing = glob_mod.glob(gallery_pattern)
        if existing:
            continue

        # Mark that we need to scan this game's gallery
        marker = cache / f"{bgg_id}_gallery_scanned.marker"
        if marker.exists():
            continue  # already scanned, found nothing

        to_fetch.append({"name": name, "bgg_id": bgg_id})

    if not to_fetch:
        print(f"[image_cache] Gallery images already cached/scanned for all games")
        return

    print(f"[image_cache] Scanning gallery for {len(to_fetch)} games (up to {_GALLERY_MAX_PAGES * _GALLERY_PAGE_SIZE} images each)...")
    total_downloaded = 0
    consecutive_failures = 0

    for i, game in enumerate(to_fetch):
        gallery_paths = fetch_gallery_images(game["bgg_id"], game["name"], cache)

        if gallery_paths is None:
            # API error (rate limited, server error, etc.)
            consecutive_failures += 1
            if consecutive_failures >= 5:
                remaining = len(to_fetch) - i - 1
                print(f"[image_cache] Gallery API: {consecutive_failures} consecutive failures, "
                      f"stopping. {remaining} games skipped. Re-run to resume.")
                break
            # Exponential backoff on failures
            wait = min(30, 2 ** consecutive_failures)
            print(f"[image_cache] Gallery API error for {game['name'][:30]}, "
                  f"retrying after {wait}s ({consecutive_failures}/5 failures)")
            time.sleep(wait)
            continue

        consecutive_failures = 0  # reset on success

        if gallery_paths:
            result.setdefault(game["name"], []).extend(gallery_paths)
            total_downloaded += len(gallery_paths)
        else:
            # Write a marker so we don't re-scan games with no matches
            marker = cache / f"{game['bgg_id']}_gallery_scanned.marker"
            marker.write_text("")

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(to_fetch)}] {game['name'][:30]:30s} +{len(gallery_paths)} images")

    print(f"[image_cache] Gallery: {total_downloaded} images downloaded across {len(to_fetch)} games")


def fetch_gallery_images(
    bgg_id: int,
    game_name: str,
    cache_dir: Path,
) -> list[Path] | None:
    """
    Fetch gallery images for one game from the BGG gallery API.

    Paginates up to GALLERY_MAX_PAGES pages (500 images), scans captions
    for keywords indicating box covers/3D renders/spines, downloads matches.

    Returns:
        list[Path]: Downloaded image paths (may be empty if no caption matches)
        None: API error (rate limited, server error) — caller should handle retry
    """
    downloaded: list[Path] = []

    for page in range(1, _GALLERY_MAX_PAGES + 1):
        try:
            resp = requests.get(
                _GALLERY_API,
                params={
                    "ajax": 1,
                    "gallery": "all",
                    "nosession": 1,
                    "objectid": bgg_id,
                    "objecttype": "thing",
                    "pageid": page,
                    "showcount": _GALLERY_PAGE_SIZE,
                    "size": "thumb",
                    "sort": "hot",
                },
                timeout=_TIMEOUT,
            )
            if resp.status_code == 429:
                return None  # rate limited
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError:
            return None  # server error — signal to caller
        except Exception:
            break  # network/parse error — stop pagination but return what we have

        images = data.get("images", [])
        if not images:
            break

        for img in images:
            caption = (img.get("caption") or "").lower()
            image_id = img.get("imageid")
            image_url = img.get("imageurl_lg")

            if not image_id or not image_url:
                continue

            # Check if caption contains any of our keywords
            if not any(kw in caption for kw in _CAPTION_KEYWORDS):
                continue

            # Download this image
            dest = cache_dir / f"{bgg_id}_gallery_{image_id}.jpg"
            if dest.exists() and dest.stat().st_size > 0:
                downloaded.append(dest)
            else:
                if _download_image(image_url, dest):
                    downloaded.append(dest)

            if len(downloaded) >= _GALLERY_MAX_DOWNLOADS:
                return downloaded

        # Stop pagination if this was the last page
        if len(images) < _GALLERY_PAGE_SIZE:
            break

        time.sleep(_GALLERY_API_DELAY)

    return downloaded


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

    except Exception:
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
