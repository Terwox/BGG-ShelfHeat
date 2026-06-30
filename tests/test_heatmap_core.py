from pathlib import Path

from PIL import Image

from shelfheat.heatmap import (
    COLOR_NEVER_PLAYED,
    COLOR_NOT_IN_COLLECTION,
    COLOR_UNIDENTIFIED,
    MAX_DAYS,
    classify_item,
    compute_summary,
    days_to_color,
    generate_heatmap,
)


def test_classify_item_handles_malformed_last_played_as_ancient():
    category, color, label = classify_item(
        {"game_name": "Arcs"},
        {"name": "Arcs", "play_count": 2, "last_played": "not-a-date"},
    )

    assert category == "played_3plus"
    assert color == days_to_color(MAX_DAYS)
    assert "2 plays" in label


def test_classify_item_keeps_special_categories_distinct():
    assert classify_item(None, None) == (
        "unidentified",
        COLOR_UNIDENTIFIED,
        "Unidentified",
    )
    assert classify_item({"game_name": "Unknown"}, None) == (
        "not_in_collection",
        COLOR_NOT_IN_COLLECTION,
        "Not in collection",
    )
    assert classify_item(
        {"game_name": "Unplayed"},
        {"name": "Unplayed", "play_count": 0, "last_played": None},
    ) == ("never_played", COLOR_NEVER_PLAYED, "Never played")


def test_classify_item_treats_positive_plays_without_date_as_ancient():
    assert classify_item(
        {"game_name": "Played But Undated"},
        {"name": "Played But Undated", "play_count": 2, "last_played": None},
    ) == ("played_3plus", days_to_color(MAX_DAYS), "2 plays (no date)")


def test_compute_summary_counts_contract_fields():
    items = [
        {"category": "played_recent", "identification": {"game_name": "A"}, "collection_match": {"name": "A"}},
        {"category": "never_played", "identification": {"game_name": "B"}, "collection_match": {"name": "B"}},
        {"category": "not_in_collection", "identification": {"game_name": "C"}, "collection_match": None},
        {"category": "unidentified", "identification": None, "collection_match": None},
    ]

    assert compute_summary(items) == {
        "total_detected": 4,
        "identified": 3,
        "matched": 2,
        "never_played": 1,
        "not_in_collection": 1,
        "unidentified": 1,
        "by_category": {
            "played_recent": 1,
            "never_played": 1,
            "not_in_collection": 1,
            "unidentified": 1,
        },
    }


def test_generate_heatmap_writes_self_contained_html(tmp_path):
    photo = tmp_path / "shelf.jpg"
    Image.new("RGB", (64, 48), color=(30, 30, 40)).save(photo)

    output = tmp_path / "shelf_heatmap.html"
    item = {
        "id": 0,
        "polygon": [[5, 5], [40, 5], [40, 30], [5, 30]],
        "identification": {"game_name": "Arcs", "method": "manual", "confidence": 1.0},
        "collection_match": {"name": "Arcs", "play_count": 3, "last_played": "2024-01-01"},
        "category": "played_1_2yr",
        "color": "#ffff00",
        "label": "Played",
    }

    result = generate_heatmap(
        photo_path=str(photo),
        items=[item],
        detection_size=(64, 48),
        output_path=str(output),
        bgg_user="alice",
        collection_games=[{"name": "Arcs", "plays": 3, "last_played": "2024-01-01"}],
    )

    html = Path(result).read_text(encoding="utf-8")
    assert output.exists()
    assert "<title>ShelfHeat" in html
    assert "data:image/jpeg;base64," in html
    assert "Arcs" in html
    assert "<polygon" in html
