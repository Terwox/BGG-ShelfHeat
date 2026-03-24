"""
CLI entry point: orchestrate the full pipeline.

    photo → detect → segment → crop → identify → match → heatmap

Usage:
    shelfheat photo.jpg --bgg-user alice
    shelfheat shelf1.jpg shelf2.jpg --collection my_games.csv --output ./results
    python -m shelfheat photo.jpg --bgg-user alice
"""

import argparse
import json
import sys
import time
import html as html_mod
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for Unicode box-drawing characters
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from shelfheat import __version__
from shelfheat.detect import detect_boxes
from shelfheat.heatmap import classify_item, compute_summary, generate_heatmap
from shelfheat.identify import GameIdentifier, polygon_crop
from shelfheat.image_cache import ensure_collection_images, enrich_from_bggdb
from shelfheat.match import BGGCollection
from shelfheat.segment import segment_boxes


def main():
    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load BGG collection (shared across all photos) ──────────────
    collection = _load_collection(args)
    game_names = collection.game_names()
    print(f"\n{'='*60}")
    print(f"Collection: {len(game_names)} games")
    print(f"{'='*60}\n")

    # ── Download/cache box art images for CLIP image matching ─────
    game_images = {}
    if not getattr(args, "no_images", False):
        enrich_from_bggdb(collection.games)
        game_images = ensure_collection_images(collection.games)
    else:
        print("[pipeline] Skipping box art image download (--no-images)")

    use_tiling = not getattr(args, "no_tiling", False)

    # ── Process each photo ──────────────────────────────────────────
    for photo_path in args.photos:
        photo = Path(photo_path)
        if not photo.exists():
            print(f"[error] File not found: {photo}")
            continue

        print(f"\n{'─'*60}")
        print(f"Processing: {photo.name}")
        print(f"{'─'*60}")
        t0 = time.time()

        results = _run_pipeline(str(photo), collection, game_names, game_images, use_tiling)
        elapsed = time.time() - t0

        # Write heatmap HTML
        stem = photo.stem
        html_path = output_dir / f"{stem}_heatmap.html"
        # Full collection for the edit UI (name + play data)
        collection_for_ui = [
            {
                "name": html_mod.unescape(g["name"]),
                "plays": g.get("play_count", 0),
                "last_played": g.get("last_played") or "",
            }
            for g in collection.games if g.get("name")
        ]

        generate_heatmap(
            photo_path=str(photo),
            items=results["items"],
            detection_size=tuple(results["detection_size"]),
            output_path=str(html_path),
            bgg_user=args.bgg_user,
            collection_games=collection_for_ui,
        )

        # Write results JSON (for the standalone viewer)
        json_path = output_dir / f"{stem}_results.json"
        results_out = {
            "version": __version__,
            "generated": datetime.now(timezone.utc).isoformat(),
            "photo_filename": photo.name,
            "original_size": results["original_size"],
            "detection_size": results["detection_size"],
            "scale_factor": results["scale_factor"],
            "bgg_user": args.bgg_user,
            "items": results["items"],
            "summary": results["summary"],
        }
        json_path.write_text(
            json.dumps(results_out, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[results] Wrote {json_path}")

        _print_summary(results["summary"], elapsed)

    print(f"\nDone. Output in {output_dir.resolve()}/")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    photo_path: str,
    collection: BGGCollection,
    game_names: list[str],
    game_images: dict[str, "Path"] | None = None,
    tiling: bool = True,
) -> dict:
    """
    Run the full detection → identification → matching pipeline on one photo.

    Returns a dict with items (classified), sizes, scale_factor, and summary.
    """
    # Stage 1: Detect bounding boxes
    det = detect_boxes(photo_path, tiling=tiling)
    detections = det["detections"]
    scale_factor = det["scale_factor"]
    detection_size = det["detection_size"]
    original_size = det["original_size"]

    if not detections:
        print("[pipeline] No game boxes detected in this photo.")
        return {
            "items": [],
            "original_size": list(original_size),
            "detection_size": list(detection_size),
            "scale_factor": scale_factor,
            "summary": compute_summary([]),
        }

    # Stage 2: Segment into polygons
    segments = segment_boxes(photo_path, detections, scale_factor, detection_size)

    # Stage 3+4: Crop and identify each polygon
    identifier = GameIdentifier(game_names, game_images=game_images)
    items = []

    print(f"[identify] Processing {len(segments)} crops...")
    for i, seg in enumerate(segments):
        crop = polygon_crop(photo_path, seg["polygon"], scale_factor)
        ident = identifier.identify(crop)

        # Stage 5: Match identification against collection
        match_result = None
        if ident:
            match_result = collection.match(ident["game_name"])

        # Stage 6 prep: classify for heatmap coloring
        category, color, label = classify_item(ident, match_result)

        # Sanitize match_result for JSON serialization (drop numpy types)
        safe_match = _sanitize_match(match_result) if match_result else None

        items.append({
            "id": seg["id"],
            "polygon": seg["polygon"],
            "confidence": seg["confidence"],
            "sam_score": seg["sam_score"],
            "identification": ident,
            "collection_match": safe_match,
            "category": category,
            "color": color,
            "label": label,
        })

        name = ident["game_name"] if ident else "?"
        if (i + 1) % 5 == 0 or i == 0 or i == len(segments) - 1:
            print(f"  [{i+1}/{len(segments)}] {name} -> {category}")

    # Post-identification dedup: if multiple polygons identify as the
    # same game AND overlap, keep only the smallest (tightest fit).
    items = _dedup_same_game(items)

    return {
        "items": items,
        "original_size": list(original_size),
        "detection_size": list(detection_size),
        "scale_factor": scale_factor,
        "summary": compute_summary(items),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_same_game(items: list[dict]) -> list[dict]:
    """
    Remove duplicate polygons for the same game.

    Two strategies:
    1. If 3+ polygons identify as the same game, it's almost certainly
       false positives from a visually dominant reference image — keep only
       the single highest-confidence match.
    2. If exactly 2 polygons identify as the same game AND they overlap,
       keep only the smallest (tightest bounding polygon).
    """
    from shapely.geometry import Polygon

    # Group identified items by game name
    by_name: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        ident = it.get("identification")
        if ident and ident.get("game_name"):
            name = ident["game_name"].lower()
            by_name.setdefault(name, []).append(i)

    # Only process games with multiple detections
    remove_indices = set()
    for name, indices in by_name.items():
        if len(indices) < 2:
            continue

        if len(indices) >= 3:
            # Strategy 1: Too many matches — keep only highest confidence
            best_idx = max(
                indices,
                key=lambda i: items[i].get("identification", {}).get("confidence", 0),
            )
            for idx in indices:
                if idx != best_idx:
                    remove_indices.add(idx)
            continue

        # Strategy 2: Exactly 2 — check spatial overlap
        # Build Shapely polygons and compute areas
        polys = []
        for idx in indices:
            pts = items[idx]["polygon"]
            try:
                p = Polygon(pts)
                if not p.is_valid:
                    from shapely.validation import make_valid
                    p = make_valid(p)
                polys.append((idx, p, p.area))
            except Exception:
                polys.append((idx, None, float("inf")))

        # Sort by area (smallest first — tightest fit wins)
        polys.sort(key=lambda x: x[2])

        # Keep the smallest. For each larger polygon, check if it
        # overlaps with the keeper — if so, mark for removal.
        keeper_idx, keeper_poly, _ = polys[0]
        for idx, poly, area in polys[1:]:
            if poly is None or keeper_poly is None:
                remove_indices.add(idx)
                continue
            try:
                if poly.intersects(keeper_poly):
                    remove_indices.add(idx)
            except Exception:
                remove_indices.add(idx)

    if remove_indices:
        # Log what we're removing
        for idx in sorted(remove_indices):
            ident = items[idx].get("identification") or {}
            name = ident.get("game_name", "?")
            print(f"  [dedup] Removing oversized duplicate: {name} (id={items[idx]['id']})")

        items = [it for i, it in enumerate(items) if i not in remove_indices]
        print(f"[dedup] Removed {len(remove_indices)} same-game duplicates")

    return items


def _load_collection(args: argparse.Namespace) -> BGGCollection:
    """Load BGG collection from API or CSV based on CLI args."""
    if args.bgg_user:
        return BGGCollection.from_api(args.bgg_user)
    return BGGCollection.from_csv(args.collection)


def _sanitize_match(match: dict) -> dict:
    """Ensure all values are JSON-serializable (no numpy scalars)."""
    clean = {}
    for k, v in match.items():
        if hasattr(v, "item"):  # numpy scalar
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


def _print_summary(summary: dict, elapsed: float):
    """Print a human-readable summary to stdout."""
    s = summary
    print(f"\n  {'='*40}")
    print(f"  Detected:         {s['total_detected']}")
    print(f"  Identified:       {s['identified']}")
    print(f"  Matched:          {s['matched']}")
    print(f"  Never played:     {s['never_played']}")
    print(f"  Not in collection:{s['not_in_collection']}")
    print(f"  Unidentified:     {s['unidentified']}")

    by_cat = s.get("by_category", {})
    played_cats = [k for k in by_cat if k.startswith("played_")]
    if played_cats:
        print(f"  ──────────────────")
        for cat in sorted(played_cats):
            nice = cat.replace("played_", "").replace("_", " ")
            print(f"  Played ({nice}): {by_cat[cat]}")

    print(f"  {'='*40}")
    print(f"  Elapsed: {elapsed:.1f}s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="shelfheat",
        description="Board game shelf photo → play-recency heatmap using BGG data.",
        epilog="Example: shelfheat shelf.jpg --bgg-user alice",
    )
    p.add_argument(
        "photos",
        nargs="+",
        help="Path(s) to shelf photo(s)",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--bgg-user",
        metavar="USERNAME",
        help="Fetch collection from BoardGameGeek for this username",
    )
    src.add_argument(
        "--collection",
        metavar="CSV_PATH",
        help="Path to a BGG collection CSV export",
    )

    p.add_argument(
        "--output", "-o",
        default="output",
        metavar="DIR",
        help="Output directory (default: ./output)",
    )
    p.add_argument(
        "--no-tiling",
        action="store_true",
        default=False,
        help="Disable tiled multi-scale detection (faster, less accurate)",
    )
    p.add_argument(
        "--no-images",
        action="store_true",
        default=False,
        help="Skip downloading BGG box art images (disables CLIP image matching)",
    )
    p.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = p.parse_args()

    # Validate photo paths early
    for photo in args.photos:
        if not Path(photo).exists():
            p.error(f"Photo not found: {photo}")

    return args


if __name__ == "__main__":
    main()
