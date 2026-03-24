"""Full end-to-end pipeline test including heatmap output."""
import sys, traceback
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

try:
    from shelfheat.cli import main
    sys.argv = [
        "shelfheat",
        r"D:\git\LessBoardGames\static\shelves\13b17c3b-2de1-4d1c-aaa5-58d1effa3e3f.jpg",
        "--collection", r"D:\git\BGG-ShelfHeat\test-collection.csv",
        "--output", r"D:\git\BGG-ShelfHeat\output",
    ]
    main()
except Exception as e:
    print(f"\n\n=== CRASH ===")
    traceback.print_exc()
