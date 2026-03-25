"""Batch-process all Alice's shelf photos through ShelfHeat pipeline."""
import subprocess
import sys
import time
from pathlib import Path

SHELF_DIR = Path(r"D:\git\LessBoardGames\static\shelves")
OUTPUT_DIR = Path(r"D:\git\BGG-ShelfHeat\output")
COLLECTION = Path(r"D:\git\BGG-ShelfHeat\test-collection.csv")

# All photos minus dupes (5fc07c02, b5344b10) and already-done closet (13b17c3b)
PHOTOS = [
    "72c36b40-3615-43a5-b03b-496206385dfb.jpg",   # Primary shelves
    "8f049fc6-f7f4-48c1-82b7-5a10fdeb1b82.jpg",   # Primary shelves (cropped)
    "084b09f9-7877-4cee-8985-dfd96751726b.jpg",   # Bedroom shelves
    "5b5361f5-bf41-4cc6-903a-6ecbdca7a2d4.jpg",   # Office floor corner
    "5228a42d-741a-44a2-9bce-f4aef8734aa5.jpg",   # Game room metal (left)
    "acfc938c-4ddf-49a2-bd76-515feed5fd0f.jpg",   # Game room metal (right)
    "5bc872fd-c1af-4337-8540-00781c320971.jpg",   # Game room corner
    "1005cab7-5a7d-4006-b56d-7b2116b6e009.jpg",   # Hallway closet
    "1d644fad-ef0b-4cdc-a959-ddc372366fa6.jpg",   # Hallway floor stacks
    "23f42dcc-162f-4af6-bb4a-5d0f11202659.jpg",   # Bottom shelf close-up
    "1b65d623-2740-4b76-837d-0314d6eb75ba.jpg",   # Living room (left)
    "c555fa97-3a30-4ac1-a7d3-0b1be701c302.jpg",   # Living room (right)
]

sys.stdout.reconfigure(encoding="utf-8")

total_start = time.time()
for i, fname in enumerate(PHOTOS, 1):
    photo = SHELF_DIR / fname
    pid = fname[:8]
    heatmap = OUTPUT_DIR / f"{fname.replace('.jpg', '')}_heatmap.html"
    
    if heatmap.exists():
        print(f"[{i}/{len(PHOTOS)}] {pid} — already done, skipping")
        continue
    
    print(f"\n{'='*60}")
    print(f"[{i}/{len(PHOTOS)}] Processing {pid}...")
    print(f"{'='*60}")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "shelfheat", str(photo),
         "--collection", str(COLLECTION),
         "--output", str(OUTPUT_DIR)],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"✅ {pid} done in {elapsed:.1f}s")
    else:
        print(f"❌ {pid} FAILED (exit {result.returncode}) in {elapsed:.1f}s")

total = time.time() - total_start
print(f"\n{'='*60}")
print(f"All done! {len(PHOTOS)} photos in {total:.0f}s ({total/60:.1f} min)")
print(f"Output: {OUTPUT_DIR}")
