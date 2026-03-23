"""
Stage 2: SAM2 polygon refinement.

Takes OWLv2 bounding boxes and refines them into oriented polygon masks
using SAM2 segmentation + cv2.minAreaRect.
"""

import numpy as np
import cv2
from PIL import Image

SAM2_MODEL_ID = "facebook/sam2.1-hiera-base-plus"


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

    return results
