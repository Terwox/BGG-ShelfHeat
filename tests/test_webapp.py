from pathlib import Path

import pytest
from PIL import Image

from shelfheat.cli import ShelfHeatJobResult
from shelfheat import webapp


def test_run_web_job_uses_username_flow_and_removes_prepared_upload(
    tmp_path,
    monkeypatch,
):
    photo = tmp_path / "shelf.png"
    Image.new("RGB", (64, 48), color=(20, 30, 40)).save(photo)
    captured = {}

    def fake_run_shelfheat_job(photos, **kwargs):
        captured["photos"] = [Path(p) for p in photos]
        captured["kwargs"] = kwargs
        assert captured["photos"][0].exists()
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "shelfheat_upload_heatmap.html"
        json_path = output_dir / "shelfheat_upload_results.json"
        html_path.write_text("<title>ShelfHeat</title>", encoding="utf-8")
        json_path.write_text("{}", encoding="utf-8")
        return [
            ShelfHeatJobResult(
                photo_path=captured["photos"][0],
                html_path=html_path,
                json_path=json_path,
                summary={"total_detected": 1, "identified": 1, "matched": 1},
                elapsed=1.25,
            )
        ]

    monkeypatch.setattr(webapp, "run_shelfheat_job", fake_run_shelfheat_job)
    monkeypatch.setattr(webapp, "WEB_TMP_ROOT", tmp_path / "webtmp")

    result = webapp.run_web_job(
        photo,
        "alice",
        max_upload_bytes=1024 * 1024,
        max_image_pixels=10_000,
    )

    assert result.html_path.exists()
    assert result.json_path.exists()
    assert captured["kwargs"]["bgg_user"] == "alice"
    assert captured["kwargs"]["no_images"] is True
    assert captured["kwargs"]["no_gallery"] is True
    assert captured["kwargs"]["inherit_plays"] is False
    assert not captured["photos"][0].exists()


def test_run_web_job_rejects_missing_username(tmp_path):
    photo = tmp_path / "shelf.jpg"
    Image.new("RGB", (16, 16)).save(photo)

    with pytest.raises(ValueError, match="BGG username"):
        webapp.run_web_job(photo, " ")


def test_run_web_job_rejects_non_image_upload(tmp_path):
    upload = tmp_path / "notes.txt"
    upload.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="JPEG or PNG|readable"):
        webapp.run_web_job(upload, "alice")


def test_run_web_job_removes_job_dir_on_pipeline_failure(tmp_path, monkeypatch):
    photo = tmp_path / "shelf.png"
    Image.new("RGB", (64, 48), color=(20, 30, 40)).save(photo)
    web_tmp = tmp_path / "webtmp"

    def fail_run_shelfheat_job(*args, **kwargs):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(webapp, "run_shelfheat_job", fail_run_shelfheat_job)
    monkeypatch.setattr(webapp, "WEB_TMP_ROOT", web_tmp)

    with pytest.raises(RuntimeError, match="pipeline failed"):
        webapp.run_web_job(
            photo,
            "alice",
            max_upload_bytes=1024 * 1024,
            max_image_pixels=10_000,
        )

    assert list(web_tmp.iterdir()) == []


def test_run_web_job_removes_job_dir_when_pipeline_returns_no_results(tmp_path, monkeypatch):
    photo = tmp_path / "shelf.png"
    Image.new("RGB", (64, 48), color=(20, 30, 40)).save(photo)
    web_tmp = tmp_path / "webtmp"

    monkeypatch.setattr(webapp, "run_shelfheat_job", lambda *args, **kwargs: [])
    monkeypatch.setattr(webapp, "WEB_TMP_ROOT", web_tmp)

    with pytest.raises(RuntimeError, match="did not produce"):
        webapp.run_web_job(
            photo,
            "alice",
            max_upload_bytes=1024 * 1024,
            max_image_pixels=10_000,
        )

    assert list(web_tmp.iterdir()) == []


def test_run_web_job_removes_job_dir_on_prepare_failure(tmp_path, monkeypatch):
    photo = tmp_path / "shelf.png"
    Image.new("RGB", (64, 48), color=(20, 30, 40)).save(photo)
    web_tmp = tmp_path / "webtmp"

    monkeypatch.setattr(webapp, "WEB_TMP_ROOT", web_tmp)

    with pytest.raises(ValueError, match="dimensions"):
        webapp.run_web_job(
            photo,
            "alice",
            max_upload_bytes=1024 * 1024,
            max_image_pixels=100,
        )

    assert list(web_tmp.iterdir()) == []


def test_gradio_run_removes_temp_upload_copy(tmp_path, monkeypatch):
    upload_root = tmp_path / "gradio"
    upload_root.mkdir()
    upload = upload_root / "upload.png"
    Image.new("RGB", (16, 16)).save(upload)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    html_path = output_dir / "heatmap.html"
    json_path = output_dir / "results.json"
    html_path.write_text("<title>ShelfHeat</title>", encoding="utf-8")
    json_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(webapp.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(webapp, "WEB_TMP_ROOT", tmp_path / "shelfheat-web")

    def fake_run_web_job(photo_path, bgg_username):
        assert Path(photo_path).exists()
        return webapp.WebJobResult(
            status="done",
            html_path=html_path,
            json_path=json_path,
            summary={},
        )

    monkeypatch.setattr(webapp, "run_web_job", fake_run_web_job)

    status, preview, html_file, json_file = webapp._gradio_run(str(upload), "alice")

    assert status == "done"
    assert "ShelfHeat" in preview
    assert html_file == str(html_path)
    assert json_file == str(json_path)
    assert not upload.exists()


def test_webapp_source_has_no_collection_file_prompt():
    source = Path(webapp.__file__).read_text(encoding="utf-8").lower()

    assert "collection csv" not in source
    assert "bgg export" not in source
