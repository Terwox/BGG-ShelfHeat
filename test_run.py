"""Test run of the pipeline on one photo with error tracing."""
import sys, traceback
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from shelfheat.detect import detect_boxes
from shelfheat.segment import segment_boxes
from shelfheat.identify import GameIdentifier, polygon_crop
from shelfheat.match import BGGCollection

PHOTO = r"D:\git\LessBoardGames\static\shelves\13b17c3b-2de1-4d1c-aaa5-58d1effa3e3f.jpg"
CSV = r"D:\git\BGG-ShelfHeat\test-collection.csv"

# Load collection
coll = BGGCollection.from_csv(CSV)
names = coll.game_names()
print(f"Collection: {len(names)} games")

# Detect
det = detect_boxes(PHOTO)
dets = det["detections"]
sf = det["scale_factor"]
dsz = det["detection_size"]
print(f"Detected: {len(dets)} boxes")

# Segment
segs = segment_boxes(PHOTO, dets, sf, dsz)
print(f"Segmented: {len(segs)} polygons")

# Identify crops one by one with error handling
identifier = GameIdentifier(names)
results = []
for i, seg in enumerate(segs):
    try:
        crop = polygon_crop(PHOTO, seg["polygon"], sf)
        if crop is None or crop.size == 0:
            print(f"  [{i}] empty crop, skipping")
            results.append(None)
            continue
        result = identifier.identify(crop)
        name = result["game_name"] if result else "?"
        method = result["method"] if result else "-"
        conf = result["confidence"] if result else 0
        if i < 10 or result:
            print(f"  [{i}] {method}:{conf:.2f} -> {name}")
        results.append(result)
    except Exception as e:
        print(f"  [{i}] CRASH: {e}")
        traceback.print_exc()
        results.append(None)

identified = sum(1 for r in results if r)
print(f"\nIdentified: {identified}/{len(segs)}")
print("Test complete.")
