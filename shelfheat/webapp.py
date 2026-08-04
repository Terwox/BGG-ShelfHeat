"""Gradio web app wrapper for ShelfHeat."""

from __future__ import annotations

import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from shelfheat.cli import run_shelfheat_job
from shelfheat.match import BGGApiError, BGGConfigurationError, BGGUnavailableError

WEB_TMP_ROOT = Path(tempfile.gettempdir()) / "shelfheat-web"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 6_000_000
MAX_SOURCE_PIXELS = 4 * MAX_IMAGE_PIXELS
JOB_TTL_SECONDS = 60 * 60
USERNAME_RE = re.compile(r"^[A-Za-z0-9_. -]{1,64}$")

Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS


@dataclass(frozen=True)
class WebJobResult:
    status: str
    html_path: Path
    json_path: Path
    summary: dict


def run_web_job(
    photo_path: str | Path,
    bgg_username: str,
    *,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
    job_ttl_seconds: int = JOB_TTL_SECONDS,
) -> WebJobResult:
    """Run one public web job using conservative network and storage defaults."""
    username = _validate_username(bgg_username)
    source = Path(photo_path)
    _validate_upload(source, max_upload_bytes=max_upload_bytes)

    cleanup_old_jobs(max_age_seconds=job_ttl_seconds)
    job_dir = _new_job_dir()
    prepared_photo: Path | None = None

    try:
        prepared_photo = _prepare_image(
            source,
            job_dir,
            max_image_pixels=max_image_pixels,
        )
        results = run_shelfheat_job(
            [prepared_photo],
            output_dir=job_dir / "outputs",
            bgg_user=username,
            no_images=True,
            no_gallery=True,
            no_tiling=False,
            inherit_plays=False,
        )
        if not results:
            raise RuntimeError("ShelfHeat did not produce any output.")
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        if prepared_photo is not None:
            prepared_photo.unlink(missing_ok=True)

    result = results[0]
    status = _summary_markdown(username, result.summary, result.elapsed)
    return WebJobResult(
        status=status,
        html_path=result.html_path,
        json_path=result.json_path,
        summary=result.summary,
    )


def cleanup_old_jobs(
    *,
    max_age_seconds: int = JOB_TTL_SECONDS,
    now: float | None = None,
) -> None:
    """Remove old web-job output folders from the temporary directory."""
    root = WEB_TMP_ROOT
    if not root.exists():
        return

    cutoff = (time.time() if now is None else now) - max_age_seconds
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def create_demo():
    """Build the Gradio Blocks app lazily so tests need not import Gradio."""
    import gradio as gr

    with gr.Blocks(
        title="ShelfHeat",
    ) as demo:
        gr.Markdown(
            "# ShelfHeat\n"
            "Upload a shelf photo, enter a BGG username, and get a play-recency heatmap."
        )
        with gr.Row():
            photo = gr.File(
                label="Shelf photo",
                file_types=["image"],
                type="filepath",
            )
            username = gr.Textbox(
                label="BGG username",
                max_lines=1,
                placeholder="terwox",
            )

        run_button = gr.Button("Run ShelfHeat", variant="primary")
        status = gr.Markdown()
        preview = gr.HTML()
        with gr.Row():
            html_file = gr.File(label="Heatmap HTML")
            json_file = gr.File(label="Results JSON")

        gr.HTML(
            '<p class="bgg-attribution">'
            'Powered by <a href="https://boardgamegeek.com" target="_blank" '
            'rel="noopener noreferrer">BoardGameGeek</a>. '
            "Generated downloads are short-lived and include your uploaded "
            "photo plus collection-derived data so the heatmap works offline."
            "</p>"
        )

        run_button.click(
            fn=_gradio_run,
            inputs=[photo, username],
            outputs=[status, preview, html_file, json_file],
        )

    return demo.queue(default_concurrency_limit=1, max_size=8)


def _gradio_run(photo_path: str | None, bgg_username: str):
    if not photo_path:
        return ("Please upload a shelf photo.", "", None, None)

    try:
        result = run_web_job(photo_path, bgg_username)
    except BGGConfigurationError:
        return (
            "ShelfHeat is missing its server-side BGG API configuration.",
            "",
            None,
            None,
        )
    except BGGUnavailableError as exc:
        return (str(exc), "", None, None)
    except BGGApiError as exc:
        return (str(exc), "", None, None)
    except ValueError as exc:
        return (str(exc), "", None, None)
    except Exception:
        return (
            "ShelfHeat could not finish this job. Please try a smaller, clearer photo.",
            "",
            None,
            None,
        )
    finally:
        _cleanup_gradio_upload(photo_path)

    preview_html = result.html_path.read_text(encoding="utf-8")
    return (
        result.status,
        preview_html,
        str(result.html_path),
        str(result.json_path),
    )


def _validate_username(raw: str) -> str:
    username = (raw or "").strip()
    if not username:
        raise ValueError("Enter a BGG username.")
    if not USERNAME_RE.fullmatch(username):
        raise ValueError("Use a plain BGG username without special characters.")
    return username


def _validate_upload(path: Path, *, max_upload_bytes: int) -> None:
    if not path.exists():
        raise ValueError("Uploaded photo was not found.")
    if path.stat().st_size > max_upload_bytes:
        raise ValueError("Photo is too large for the free web demo.")


def _cleanup_gradio_upload(photo_path: str | Path) -> None:
    path = Path(photo_path)
    try:
        resolved = path.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        web_tmp = WEB_TMP_ROOT.resolve()
    except OSError:
        return

    if not _is_relative_to(resolved, temp_root):
        return
    if _is_relative_to(resolved, web_tmp):
        return

    try:
        resolved.unlink(missing_ok=True)
    except OSError:
        return


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _new_job_dir() -> Path:
    WEB_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = WEB_TMP_ROOT / f"job-{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True)
    return job_dir


def _prepare_image(
    source: Path,
    job_dir: Path,
    *,
    max_image_pixels: int,
) -> Path:
    try:
        with Image.open(source) as img:
            fmt = (img.format or "").upper()
            if fmt not in {"JPEG", "PNG"}:
                raise ValueError("Upload a JPEG or PNG shelf photo.")

            source_pixels = img.width * img.height
            if source_pixels > max_image_pixels * 4:
                raise ValueError("Photo dimensions are too large for the free web demo.")

            target_size = None
            if source_pixels > max_image_pixels:
                scale = (max_image_pixels / source_pixels) ** 0.5
                size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                target_size = size
                img.draft("RGB", size)

            img = ImageOps.exif_transpose(img)
            img = img.copy()
            if target_size:
                img.thumbnail(target_size, Image.Resampling.LANCZOS)

            prepared = job_dir / "shelfheat_upload.jpg"
            img.convert("RGB").save(prepared, "JPEG", quality=90)
            return prepared
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Upload a readable JPEG or PNG shelf photo.") from exc


def _summary_markdown(username: str, summary: dict, elapsed: float) -> str:
    payload = {
        "BGG user": username,
        "Detected": summary.get("total_detected", 0),
        "Identified": summary.get("identified", 0),
        "Matched": summary.get("matched", 0),
        "Never played": summary.get("never_played", 0),
        "Elapsed seconds": round(elapsed, 1),
    }
    lines = ["### Done", ""]
    for key, value in payload.items():
        lines.append(f"- **{key}:** {value}")
    return "\n".join(lines)
