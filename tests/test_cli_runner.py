import json
import sys
import types
from pathlib import Path

from shelfheat import cli


class FakeCollection:
    def __init__(self):
        self.games = [
            {
                "name": "Arcs",
                "bgg_id": 123,
                "play_count": 2,
                "last_played": "2024-06-01",
            }
        ]
        self.match_calls = []

    def game_names(self):
        return ["Arcs"]

    def match(self, query, inherit_plays=True):
        self.match_calls.append((query, inherit_plays))
        return self.games[0]


def test_run_pipeline_can_disable_related_play_inheritance(monkeypatch):
    collection = FakeCollection()

    monkeypatch.setitem(
        sys.modules,
        "shelfheat.detect",
        types.SimpleNamespace(
            detect_boxes=lambda *args, **kwargs: {
                "detections": [{"bbox": [0, 0, 10, 10]}],
                "scale_factor": 1,
                "detection_size": [100, 80],
                "original_size": [100, 80],
            }
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "shelfheat.segment",
        types.SimpleNamespace(
            segment_boxes=lambda *args, **kwargs: [
                {
                    "id": 0,
                    "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
                    "confidence": 0.9,
                    "sam_score": 0.8,
                }
            ]
        ),
    )

    class FakeIdentifier:
        def __init__(self, *args, **kwargs):
            pass

        def identify(self, crop):
            return {"game_name": "Arcs", "confidence": 0.95}

    monkeypatch.setitem(
        sys.modules,
        "shelfheat.identify",
        types.SimpleNamespace(
            GameIdentifier=FakeIdentifier,
            polygon_crop=lambda *args, **kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "shelfheat.heatmap",
        types.SimpleNamespace(
            classify_item=lambda ident, match: ("played_recent", "#00ff00", "Played"),
            compute_summary=lambda items: {
                "total_detected": len(items),
                "identified": len(items),
                "matched": len(items),
            },
        ),
    )

    result = cli._run_pipeline(
        "photo.jpg",
        collection,
        ["Arcs"],
        game_images=None,
        tiling=True,
        inherit_plays=False,
    )

    assert result["summary"]["matched"] == 1
    assert collection.match_calls == [("Arcs", False)]


def test_run_shelfheat_job_web_defaults_skip_image_cache_and_write_outputs(
    tmp_path,
    monkeypatch,
):
    photo = tmp_path / "shelf.jpg"
    photo.write_bytes(b"not-used-by-mocked-pipeline")
    collection = FakeCollection()

    monkeypatch.setattr(
        cli,
        "load_collection",
        lambda **kwargs: collection,
    )

    def fake_run_pipeline(*args, **kwargs):
        assert kwargs["inherit_plays"] is False
        return {
            "items": [],
            "original_size": [100, 80],
            "detection_size": [100, 80],
            "scale_factor": 1,
            "summary": {
                "total_detected": 0,
                "identified": 0,
                "matched": 0,
            },
        }

    monkeypatch.setattr(cli, "_run_pipeline", fake_run_pipeline)

    def fake_generate_heatmap(**kwargs):
        output = Path(kwargs["output_path"])
        output.write_text("<title>ShelfHeat</title>", encoding="utf-8")
        return str(output)

    monkeypatch.setitem(
        sys.modules,
        "shelfheat.heatmap",
        types.SimpleNamespace(generate_heatmap=fake_generate_heatmap),
    )
    monkeypatch.setitem(
        sys.modules,
        "shelfheat.image_cache",
        types.SimpleNamespace(
            ensure_collection_images=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("image cache should not run")
            ),
            enrich_from_bggdb=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("image enrichment should not run")
            ),
        ),
    )

    results = cli.run_shelfheat_job(
        [photo],
        output_dir=tmp_path / "out",
        bgg_user="alice",
        no_images=True,
        inherit_plays=False,
    )

    assert len(results) == 1
    assert results[0].html_path.exists()
    assert results[0].json_path.exists()
    payload = json.loads(results[0].json_path.read_text(encoding="utf-8"))
    assert payload["bgg_user"] == "alice"
