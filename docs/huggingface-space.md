# Hugging Face Space Deployment

ShelfHeat's public beta is designed for a Hugging Face Gradio Space on free CPU Basic hardware.

## Space Settings

Use this Space README front matter in the Space repository:

```yaml
---
title: ShelfHeat
emoji: 🎲
colorFrom: green
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
license: mit
---
```

## Required Secret

Configure this secret in the Space settings:

```text
BGG_API_TOKEN
```

Do not commit the token. Local development may use `.secrets`; Hugging Face should use the Space secret.

## Publish Command

The repo includes a deployment helper that creates or updates the Space, adds the
`BGG_API_TOKEN` Space secret, and uploads only release-safe files:

```powershell
uv run python scripts/deploy-hf-space.py --private
```

The helper reads `BGG_API_TOKEN` and `HF_TOKEN` from the environment or local
`.secrets`; it never prints the token. It defaults to `<token owner>/shelfheat`; pass an
explicit repo id, such as `YOUR_HF_USERNAME/shelfheat`, to publish elsewhere.
Use `--dry-run` to inspect the upload allowlist before publishing.

## Free CPU Defaults

- One queued job at a time through Gradio queueing.
- JPEG/PNG upload validation with a 20 MB file limit.
- Oversized images are downscaled before inference.
- Web jobs run with `no_images=True`.
- Web jobs disable related-game play inheritance so matching does not make request-time BGG `/thing` calls.
- Per-user uploads are removed after processing; generated output files live only in a temporary job directory until cleanup.

## Release Smoke Test

Before posting to BGG:

1. Open the public Space in a clean browser session.
2. Upload a small non-private shelf photo.
3. Enter a public BGG username.
4. Confirm BGG collection/play fetch succeeds with the server-side token.
5. Confirm the job returns downloadable HTML and JSON.
6. Confirm the generated HTML supports hover and click-to-edit.
7. Trigger one invalid photo and one invalid BGG username.
8. Confirm logs do not show token values or raw collection payloads.
9. Record cold-start and warm-run durations in release notes.

## Docker Fallback

Start with Gradio SDK. Switch to Docker only if managed Gradio fails on system dependencies for OpenCV, EasyOCR, or model runtime packages.
