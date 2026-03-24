"""
Stage 2: SAM2 polygon refinement + overlap suppression.

Takes OWLv2 bounding boxes and refines them into oriented polygon masks
using SAM2 segmentation + cv2.minAreaRect, then removes physical overlaps
via Shapely polygon clipping (painter's algorithm — highest confidence first).
"""

import numpy as np
import cv2
from PIL import Image
from shapely.geometry import Polygon
from shapely.validation import make_valid

SAM2_MODEL_ID = "facebook/sam2.1-hiera-base-plus"

# Overlap suppression config
MIN_AREA_RATIO = 0.20  # discard if clipped below 20% of original area
MIN_ABSOLUTE_AREA = 100  # discard tiny polygons (in detection-res pixels²)
MAX_AREA_MULTIPLIER = 5.0  # discard if > 5× median area (shelf segments, not game boxes)


def segment_boxes(
    image_path: str,
    detections: list[dict],
    scale_factor: float,
    detection_size: tuple[int, int],
) -> list[dict]:
    """
    Refine axis-aligned boxes into oriented polygons via SAM2.

    Args:
        image_path:     Path to the original photo.
        detections:     List from detect.py (each has "bbox" in detection res).
        scale_factor:   From detect.py — detection_res / original_res.
        detection_size: (w, h) of the detection-resolution image.

    Returns list of dicts, each with:
        id, confidence, sam_score, polygon (4 corner points in detection res),
        center, size, angle, rotated (bool)
    """
    import torch
    from transformers import Sam2Processor, Sam2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[segment] Loading SAM2 on {device}...")
    processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(device)
    model.eval()

    # Resize image to detection resolution (same size OWLv2 saw)
    det_w, det_h = detection_size
    image = Image.open(image_path).convert("RGB").resize((det_w, det_h), Image.LANCZOS)

    results = []
    n = len(detections)
    print(f"[segment] Refining {n} detections...")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]

        # SAM2 expects input_boxes as [[[x1, y1, x2, y2]]]
        inputs = processor(
            images=image,
            input_boxes=[[[x1, y1, x2, y2]]],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # pred_masks: (batch, num_boxes, num_masks, H, W)
        # iou_scores: (batch, num_boxes, num_masks)
        masks = outputs.pred_masks.cpu().float()
        iou_scores = outputs.iou_scores.cpu().float()

        best_idx = iou_scores[0, 0].argmax().item()
        mask = masks[0, 0, best_idx]
        sam_score = iou_scores[0, 0, best_idx].item()

        # Resize mask to detection dimensions if needed
        mh, mw = mask.shape
        if mh != det_h or mw != det_w:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(det_h, det_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        mask_u8 = (mask > 0.5).numpy().astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback: treat original bbox as polygon
            results.append(
                {
                    "id": det["id"],
                    "confidence": det["confidence"],
                    "sam_score": round(sam_score, 4),
                    "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
                    "size": [round(x2 - x1, 1), round(y2 - y1, 1)],
                    "angle": 0.0,
                    "rotated": False,
                }
            )
            continue

        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
        box_pts = cv2.boxPoints(rect)

        polygon = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in box_pts]
        results.append(
            {
                "id": det["id"],
                "confidence": det["confidence"],
                "sam_score": round(sam_score, 4),
                "polygon": polygon,
                "center": [round(float(rect[0][0]), 1), round(float(rect[0][1]), 1)],
                "size": [round(float(rect[1][0]), 1), round(float(rect[1][1]), 1)],
                "angle": round(float(rect[2]), 1),
                "rotated": True,
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n}] id={det['id']} angle={rect[2]:.1f} sam={sam_score:.3f}")

    rotated = sum(1 for r in results if r["rotated"])
    print(f"[segment] {rotated}/{len(results)} refined to oriented polygons")

    del model, processor
    if device == "cuda":
        torch.cuda.empty_cache()

    # Post-process: remove physical overlaps
    results = suppress_overlaps(results)

    return results


# ---------------------------------------------------------------------------
# Overlap suppression (painter's algorithm with Shapely)
# ---------------------------------------------------------------------------

def suppress_overlaps(segments: list[dict]) -> list[dict]:
    """
    Remove overlapping polygon regions using a greedy painter's algorithm.

    Highest-confidence polygons claim their area first. Lower-confidence
    polygons get clipped to remove any overlap. If clipping shrinks a
    polygon below MIN_AREA_RATIO of its original size, it's discarded.

    Physical constraint: game boxes can't overlap in the camera's 2D projection.
    """
    if not segments:
        return segments

    # Sort by confidence (highest first — they get priority)
    ordered = sorted(segments, key=lambda s: s["confidence"], reverse=True)

    # --- Phase 1: Area outlier filter ---
    # Compute polygon areas and reject anything > MAX_AREA_MULTIPLIER × median
    areas = []
    for seg in ordered:
        pts = seg["polygon"]
        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = make_valid(poly)
            areas.append(poly.area)
        except Exception:
            areas.append(0)

    positive_areas = [a for a in areas if a > 0]
    if positive_areas:
        import statistics
        median_area = statistics.median(positive_areas)
        max_area = median_area * MAX_AREA_MULTIPLIER
        area_rejected = 0
        filtered = []
        for seg, area in zip(ordered, areas):
            if area > max_area:
                area_rejected += 1
            else:
                filtered.append(seg)
        if area_rejected:
            print(f"[segment] Area filter: removed {area_rejected} oversized polygons (>{MAX_AREA_MULTIPLIER}× median)")
        ordered = filtered

    # --- Phase 2: Overlap clipping ---
    claimed = []       # Shapely polygons that have claimed space
    surviving = []     # segments that survive suppression

    before = len(ordered)

    for seg in ordered:
        pts = seg["polygon"]
        if len(pts) < 3:
            continue

        try:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = make_valid(poly)
            if poly.is_empty or poly.area < MIN_ABSOLUTE_AREA:
                continue
        except Exception:
            continue

        original_area = poly.area

        # Subtract all previously claimed regions
        for claimed_poly in claimed:
            if poly.intersects(claimed_poly):
                try:
                    poly = poly.difference(claimed_poly)
                    if not poly.is_valid:
                        poly = make_valid(poly)
                except Exception:
                    break

        if poly.is_empty:
            continue

        remaining_ratio = poly.area / original_area if original_area > 0 else 0

        if remaining_ratio < MIN_AREA_RATIO:
            continue  # too much was clipped away — probably a duplicate

        if poly.area < MIN_ABSOLUTE_AREA:
            continue

        # Update polygon to the clipped version
        # If clipping produced a MultiPolygon, keep the largest piece
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)

        # Convert back to our polygon format (4-point oriented box)
        clipped_pts = _shapely_to_oriented_box(poly)
        seg_copy = {**seg}
        seg_copy["polygon"] = clipped_pts
        seg_copy["clipped"] = remaining_ratio < 0.95  # flag if it was trimmed
        seg_copy["clip_ratio"] = round(remaining_ratio, 3)

        surviving.append(seg_copy)
        claimed.append(Polygon(pts))  # claim the ORIGINAL area, not the clipped one

    after = len(surviving)
    removed = before - after
    clipped = sum(1 for s in surviving if s.get("clipped", False))
    print(f"[segment] Overlap suppression: {before} → {after} ({removed} removed, {clipped} clipped)")

    return surviving


def _shapely_to_oriented_box(poly: Polygon) -> list[list[float]]:
    """Convert a Shapely polygon to a 4-point oriented bounding box."""
    # Get the minimum rotated rectangle
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)[:4]  # drop the closing duplicate
    return [[round(x, 1), round(y, 1)] for x, y in coords]
