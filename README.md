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

# BGG-ShelfHeat

Take a photo of your board game shelf. Get an interactive heatmap showing what you haven't played.

Green means "played recently." Red means "it's been a while." Purple means *shame*.

<!-- TODO: screenshot placeholder — replace with actual screenshot -->
<!-- ![ShelfHeat example](docs/screenshot.png) -->

## Try It Online

The public beta target is a free Hugging Face Space. The app flow is:

1. Upload one JPEG or PNG shelf photo.
2. Enter a BoardGameGeek username.
3. Wait while ShelfHeat fetches collection/play data server-side and runs the local vision pipeline.
4. Download the generated interactive heatmap HTML and results JSON.

Public web users do not download or upload BGG collection export files. The hosted app uses the server-side `BGG_API_TOKEN` secret and shows BoardGameGeek attribution.

Public URL: `TBD`

Deploy once `.secrets` contains `BGG_API_TOKEN` and `HF_TOKEN`:

```powershell
uv run python scripts/deploy-hf-space.py --private
```

The deploy helper reads `BGG_API_TOKEN` and `HF_TOKEN` from the environment or local `.secrets`,
creates or updates a Gradio Space, configures the Space secret, and uploads only
release-safe files. Pass an explicit repo id, such as `YOUR_HF_USERNAME/shelfheat`,
only if you do not want the helper to use the token owner's account.

## How It Works

```
Photo  →  Detect game boxes  →  Segment precise shapes  →  Identify each game  →  Match to your BGG collection  →  Color-coded heatmap
          (OWLv2)               (SAM2)                     (OCR + CLIP)           (embedding similarity)           (interactive HTML)
```

1. **Detection** — OWLv2 finds board game boxes in your shelf photo
2. **Segmentation** — SAM2 refines each box into a precise polygon outline
3. **High-res crop** — Polygons are scaled back to original photo resolution and cropped
4. **Identification** — EasyOCR reads text on spines; CLIP does visual matching if OCR can't
5. **Collection matching** — Games are matched against your BGG collection using embedding similarity (not fuzzy string matching)
6. **Heatmap** — A self-contained HTML file with your photo, colored overlays, and hover tooltips

## Setup

### Requirements

- Python 3.10+
- ~2 GB disk for model weights (downloaded automatically on first run)
- A board game shelf (non-negotiable)

### Install

```bash
# Clone the repo
git clone https://github.com/your-username/BGG-ShelfHeat.git
cd BGG-ShelfHeat

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -e .
```

### GPU vs. CPU

ShelfHeat works on both. GPU is faster, CPU is fine.

| Hardware | Time per photo |
|----------|---------------|
| NVIDIA GPU (8+ GB VRAM) | ~30 seconds |
| CPU only | ~3–5 minutes |

If you have a CUDA-capable GPU, make sure you have the right PyTorch version:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False and you have an NVIDIA GPU, reinstall PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Web app locally

Create a local `.secrets` file or set an environment variable:

```bash
BGG_API_TOKEN=your-bgg-api-token
```

Then run:

```bash
python app.py
```

The local web app uses the same conservative defaults planned for free hosting: one queued job at a time, uploaded image validation/downscaling, no BGG image/gallery downloads, and no related-game `/thing` lookups during matching.

### Basic: BGG username

```bash
shelfheat photo.jpg --bgg-user your_bgg_username
```

This fetches your collection and play history from BoardGameGeek's API, then generates the heatmap. BGG username import requires a server-side `BGG_API_TOKEN`.

### From a CSV export

If you prefer offline or the BGG API is slow:

1. Go to your [BGG collection](https://boardgamegeek.com/collection/user/YOUR_USERNAME)
2. Click the CSV export button
3. Run:

```bash
shelfheat photo.jpg --collection my_collection.csv
```

### Multiple photos

```bash
shelfheat shelf1.jpg shelf2.jpg shelf3.jpg --bgg-user alice
```

Each photo gets its own heatmap and results file.

### Custom output directory

```bash
shelfheat photo.jpg --bgg-user alice --output ./my_results
```

### Output

For each photo, you get two files in the output directory:

- **`photo_heatmap.html`** — Open in any browser. Self-contained, no server needed. Hover over games to see details.
- **`photo_results.json`** — Machine-readable results. Load this in the standalone viewer or use it in your own scripts.

## Color Key

| Color | Meaning |
|-------|---------|
| 🟢 Green | Played in the last 6 months |
| 🟡 Yellow-green | Played 6–12 months ago |
| 🟡 Yellow | Played 1–2 years ago |
| 🟠 Orange | Played 2–3 years ago |
| 🔴 Red | Played 3+ years ago |
| 🟣 Purple | Never played (shelf of shame) |
| 🔵 Blue-gray | Identified but not in your BGG collection |
| ⚪ Gray | Couldn't identify |

## Standalone Viewer

Don't want to run the pipeline again? Open `viewer/index.html` in your browser.

1. Drop (or select) your `_results.json` file
2. Drop (or select) the original shelf photo
3. Browse the interactive heatmap

No install, no server — works entirely in the browser.

## Tips for Good Results

- **Lighting matters.** Even, non-glare lighting helps OCR and CLIP a lot.
- **Straight-on angle.** Aim perpendicular to the shelf. Extreme angles make spines harder to read.
- **Resolution.** Higher is better — phone cameras work great. Webcams less so.
- **One shelf section per photo.** A single KALLAX cube or one shelf row works best. Panoramas of an entire wall will struggle.
- **Spine-out is easier.** Games with their spine facing the camera are identified more reliably than face-out covers.

## Public Beta Limitations

- Free CPU hosting is slow, especially after a cold start.
- Recognition is imperfect; glare, odd angles, stacked boxes, and tiny spines are hard.
- BGG username import depends on BGG API availability and collection visibility.
- Uploaded photos and fetched BGG collection data are processed transiently. Model caches may persist, but per-user job data should not be stored to disk beyond short-lived output downloads.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline design, model choices, and technical details.

## License

MIT
