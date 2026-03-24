"""Export the LessBoardGames collection cache to a CSV for ShelfHeat testing."""
import json, csv, sys
sys.stdout.reconfigure(encoding="utf-8")

src = r"D:\git\LessBoardGames\data\collection-cache.json"
dst = r"D:\git\BGG-ShelfHeat\test-collection.csv"

with open(src, encoding="utf-8") as f:
    data = json.load(f)

games = data["games"]
print(f"Exporting {len(games)} games from {data['username']}")

with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["objectname", "objectid", "rating", "numplays", "lastplayed", "own"])
    for g in games:
        w.writerow([
            g.get("name", ""),
            g.get("bggId", ""),
            g.get("userRating", ""),
            g.get("playCount", 0),
            g.get("lastPlayed", ""),
            1
        ])

print(f"Wrote {dst}")

# Stats
played = sum(1 for g in games if g.get("lastPlayed"))
print(f"Games with play dates: {played}/{len(games)}")
