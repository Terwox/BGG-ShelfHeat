"""
Stage 1: OWLv2 zero-shot object detection.

Supports two modes:
  - Single-pass: Resize to 1024px and detect (fast, original behavior)
  - Tiled: Run on overlapping 1024px tiles at higher resolution + a global
    context pass, then merge with cross-tile NMS (more accurate, slower)

Tracks the scale factor so downstream stages can map back to full resolution.
"""

from PIL import Image

MAX_DETECTION_DIM = 1024
MAX_WORKING_DIM = 2048
TILE_SIZE = 1024
TILE_OVERLAP = 256
TILE_STRIDE = TILE_SIZE - TILE_OVERLAP  # 768

OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
TEXT_QUERIES = [
    "board game box",
    "game box on shelf",
    "box spine",
    "board game box spine on bookshelf",
    "stacked board game boxes",
    "small card game box",
]
DEFAULT_CONFIDENCE = 0.15
DEFAULT_NMS_IOU = 0.5
CROSS_TILE_NMS_IOU = 0.3  # more aggressive for tile-boundary merging


def resize_for_detection(image: Image.Image, max_dim: int = MAX_DETECTION_DIM) -> tuple[Image.Image, float]:
    """Resize so longest edge = max_dim. Returns (resized_image, scale_factor).

    scale_factor < 1 means the image was shrunk.
    Multiply detection coords by (1/scale_factor) to get original-res coords.
    """
    w, h = image.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return resized, scale
    return image.copy(), 1.0


def detect_boxes(
    image_path: str,
    confidence: float = DEFAULT_CONFIDENCE,
    nms_iou: float = DEFAULT_NMS_IOU,
    tiling: bool = True,
) -> dict:
    """
    Detect board game boxes using OWLv2.

    Args:
        image_path: Path to the shelf photo.
        confidence: Minimum detection confidence.
        nms_iou: IoU threshold for NMS.
        tiling: If True, use tiled multi-scale detection for better accuracy.

    Returns:
        {
            "detections": [{id, label, confidence, bbox: [x1,y1,x2,y2]}],
            "scale_factor": float,       # detection_res / original_res
            "detection_size": (w, h),     # final detection-space dimensions
            "original_size": (w, h),
        }
    All bbox coordinates are in detection (resized) resolution.
    """
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[detect] Loading OWLv2 on {device}...")
    processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
    model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID).to(device)

    original = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original.size

    if tiling and max(orig_w, orig_h) > MAX_DETECTION_DIM:
        detections, det_w, det_h, scale_factor = _detect_tiled(
            original, model, processor, device, confidence, nms_iou
        )
    else:
        detections, det_w, det_h, scale_factor = _detect_single_pass(
            original, model, processor, device, confidence, nms_iou
        )

    # Re-number IDs
    for i, det in enumerate(detections):
        det["id"] = i

    print(f"[detect] {len(detections)} boxes after NMS")

    # Free VRAM
    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "detections": detections,
        "scale_factor": scale_factor,
        "detection_size": (det_w, det_h),
        "original_size": (orig_w, orig_h),
    }


# ---------------------------------------------------------------------------
# Single-pass detection (original behavior)
# ---------------------------------------------------------------------------

def _detect_single_pass(original, model, processor, device, confidence, nms_iou):
    """Run OWLv2 once on a resized image. Returns (detections, det_w, det_h, scale_factor)."""
    import torch

    resized, scale_factor = resize_for_detection(original)
    det_w, det_h = resized.size
    print(f"[detect] Single-pass: {original.size[0]}x{original.size[1]} -> {det_w}x{det_h}")

    detections = _run_owl(resized, model, processor, device, confidence)

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = _nms(detections, nms_iou)

    return detections, det_w, det_h, scale_factor


# ---------------------------------------------------------------------------
# Tiled multi-scale detection
# ---------------------------------------------------------------------------

def _detect_tiled(original, model, processor, device, confidence, nms_iou):
    """
    Run OWLv2 on overlapping tiles at working resolution + a global context pass.

    Returns (detections, det_w, det_h, scale_factor) where coordinates are in
    the working-resolution space (which becomes the "detection resolution" for
    downstream stages).
    """
    import torch

    orig_w, orig_h = original.size

    # Step 1: Resize to working resolution (up to MAX_WORKING_DIM)
    working, working_scale = resize_for_detection(original, max_dim=MAX_WORKING_DIM)
    work_w, work_h = working.size
    print(f"[detect] Tiled: {orig_w}x{orig_h} -> working {work_w}x{work_h} (scale={working_scale:.4f})")

    # Step 2: Generate tiles
    tiles = _generate_tiles(work_w, work_h)
    print(f"[detect] Generated {len(tiles)} tiles ({TILE_SIZE}px, stride {TILE_STRIDE}px)")

    # Step 3: Run OWLv2 on each tile
    all_detections = []
    for ti, (tx, ty, tw, th) in enumerate(tiles):
        tile_img = working.crop((tx, ty, tx + tw, ty + th))

        # Pad to TILE_SIZE if tile is smaller (edge tiles)
        if tw < TILE_SIZE or th < TILE_SIZE:
            padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
            padded.paste(tile_img, (0, 0))
            tile_img = padded

        tile_dets = _run_owl(tile_img, model, processor, device, confidence)

        # Map tile-local coordinates to working-resolution coordinates
        for det in tile_dets:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [
                round(x1 + tx), round(y1 + ty),
                round(x2 + tx), round(y2 + ty),
            ]
            # Clamp to working image bounds
            det["bbox"][0] = max(0, min(det["bbox"][0], work_w))
            det["bbox"][1] = max(0, min(det["bbox"][1], work_h))
            det["bbox"][2] = max(0, min(det["bbox"][2], work_w))
            det["bbox"][3] = max(0, min(det["bbox"][3], work_h))

        all_detections.extend(tile_dets)

        if (ti + 1) % 3 == 0 or ti == 0:
            print(f"  [tile {ti + 1}/{len(tiles)}] {len(tile_dets)} detections")

    print(f"[detect] {len(all_detections)} raw tile detections")

    # Step 4: Global context pass at 1024px (catches large boxes spanning tiles)
    global_img, global_scale_to_working = resize_for_detection(working, max_dim=MAX_DETECTION_DIM)
    global_dets = _run_owl(global_img, model, processor, device, confidence)

    # Map global detections back to working-resolution coordinates
    if global_scale_to_working < 1.0:
        for det in global_dets:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = [
                round(x1 / global_scale_to_working),
                round(y1 / global_scale_to_working),
                round(x2 / global_scale_to_working),
                round(y2 / global_scale_to_working),
            ]

    print(f"[detect] {len(global_dets)} global-pass detections")
    all_detections.extend(global_dets)

    # Step 5: Two-stage NMS
    # First: per-source NMS at standard threshold (already done per-tile above via _run_owl)
    # Then: cross-tile merge at more aggressive threshold
    all_detections.sort(key=lambda d: d["confidence"], reverse=True)
    merged = _nms(all_detections, CROSS_TILE_NMS_IOU)

    print(f"[detect] {len(merged)} after cross-tile NMS (IoU={CROSS_TILE_NMS_IOU})")

    # The working resolution IS our detection resolution for downstream
    return merged, work_w, work_h, working_scale


def _generate_tiles(img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """Generate overlapping tile coordinates. Returns [(x, y, w, h), ...]."""
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        th = min(TILE_SIZE, img_h - y)
        while x < img_w:
            tw = min(TILE_SIZE, img_w - x)
            tiles.append((x, y, tw, th))
            if x + TILE_SIZE >= img_w:
                break
            x += TILE_STRIDE
        if y + TILE_SIZE >= img_h:
            break
        y += TILE_STRIDE
    return tiles


# ---------------------------------------------------------------------------
# OWLv2 inference helper
# ---------------------------------------------------------------------------

def _run_owl(image, model, processor, device, confidence):
    """Run OWLv2 on a single image. Returns raw detections (no NMS)."""
    import torch

    w, h = image.size
    inputs = processor(text=[TEXT_QUERIES], images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[h, w]], device=device)
    results = processor.post_process_object_detection(
        outputs, threshold=confidence, target_sizes=target_sizes
    )[0]

    detections = []
    for i, (box, score, label_idx) in enumerate(
        zip(
            results["boxes"].cpu().tolist(),
            results["scores"].cpu().tolist(),
            results["labels"].cpu().tolist(),
        )
    ):
        x1, y1, x2, y2 = box
        detections.append(
            {
                "id": i,
                "label": TEXT_QUERIES[label_idx] if label_idx < len(TEXT_QUERIES) else "unknown",
                "confidence": round(score, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
            }
        )

    return detections


# ---------------------------------------------------------------------------
# NMS helpers
# ---------------------------------------------------------------------------

def _nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    keep = []
    for det in detections:
        if not any(_iou(det["bbox"], k["bbox"]) > iou_threshold for k in keep):
            keep.append(det)
    return keep


def _iou(a: list, b: list) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0
