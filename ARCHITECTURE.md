# BGG-ShelfHeat — Architecture

## What It Does
Take a photo of your board game shelf → get an interactive heatmap showing what you haven't played.

## Pipeline

```
Photo → [1. Detect] → [2. Segment] → [3. Crop] → [4. Identify] → [5. Match] → [6. Heatmap]
         OWLv2        SAM2           Scale-back    OCR + CLIP      BGG lookup   HTML overlay
         bounding     polygon        to full-res   constrained     embeddings   interactive
         boxes        masks          polygon crop  to game list    not fuzzy    single file
```

### Stage 1: Detection (OWLv2)
- Input: Photo resized to consistent max dimension (1024px longest edge)
- Model: `google/owlv2-base-patch16-ensemble` (local, ~600MB)
- Output: Bounding boxes with confidence scores
- Text queries: `["board game box", "game box on shelf", "box spine"]`
- Post-processing: NMS at IoU 0.5

### Stage 2: Segmentation (SAM2)
- Input: Same resized photo + OWLv2 bounding boxes as prompts
- Model: `sam2.1_hiera_base_plus` (local, ~308MB)
- Output: Precise polygon masks per detection
- `cv2.minAreaRect` → oriented bounding box with rotation angle

### Stage 3: High-Res Polygon Crop
- Scale polygon coordinates from detection resolution → original photo resolution
- Crop the bounding rect of the polygon from the ORIGINAL high-res photo
- Mask out everything outside the polygon (alpha or black fill)
- Result: Clean crop of exactly one game box, maximum resolution

### Stage 4: Identification
Tiered approach, cheapest first:

**Tier A — EasyOCR (free, local, CPU-friendly)**
- Run OCR on each polygon crop
- Extract readable text fragments
- If any fragment matches a game in the user's BGG collection → done

**Tier B — CLIP/SigLIP visual matching (free, local)**
- Encode crop image as CLIP embedding
- Compare against pre-computed text embeddings for all games in BGG collection
- Top-k nearest neighbors → candidate games

**Tier C — OpenRouter free tier (optional, 200 req/day)**
- Send crop to Qwen3-VL 235B with constrained prompt:
  "Which of these games is shown: [list of 20 nearest CLIP matches]?"
- Only used for ambiguous cases

### Stage 5: BGG Collection Matching
- Import via BGG XML API2 or user-provided CSV export
- Pre-compute text embeddings for all game names (nomic-embed-text or sentence-transformers)
- Matching uses embedding similarity, NOT SequenceMatcher
- Threshold: cosine similarity > 0.85 for auto-match, 0.7-0.85 for "needs review"
- Pull play data: playCount, lastPlayed, userRating, dateAdded

### Stage 6: Heatmap Generation
- Color gradient based on play recency:
  - Green (#00ff64): played in last 6 months
  - Yellow-green (#80ff00): 6-12 months
  - Yellow (#ffff00): 1-2 years
  - Orange (#ffa500): 2-3 years
  - Red (#ff3232): 3+ years
  - Purple (#800080): never played (shelf of shame!)
  - Blue-gray (#6464a0): not in BGG collection
  - Gray (#555): unidentified
- Output: Self-contained HTML file with embedded image + polygon data
- Interactive: hover for game details, toggle layers, click to edit

## Requirements
- Python 3.10+
- PyTorch (CPU or CUDA)
- ~2GB disk for model weights (downloaded on first run)
- A board game shelf (required)
- Shame about your unplayed games (optional but likely)

## Hardware
- **With GPU (recommended):** Full pipeline in ~30s per photo
- **CPU only:** ~3-5 min per photo. Works fine, just slower.
- **No local compute:** OpenRouter free tier only (200 identifications/day, no detection/segmentation)
