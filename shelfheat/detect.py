"""
Stage 1: OWLv2 zero-shot object detection.

Resizes input to max 1024px longest edge for consistent detection,
tracks the scale factor so downstream stages can map back to full resolution.
"""

from PIL import Image

MAX_DETECTION_DIM = 1024
OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
TEXT_QUERIES = ["board game box", "game box on shelf", "box spine"]
DEFAULT_CONFIDENCE = 0.15
DEFAULT_NMS_IOU = 0.5


def resize_for_detection(image: Image.Image) -> tuple[Image.Image, float]:
    """Resize so longest edge = MAX_DETECTION_DIM. Returns (resized_image, scale_factor).

    scale_factor < 1 means the image was shrunk.
    Multiply detection coords by (1/scale_factor) to get original-res coords.
    """
    w, h = image.size
    scale = min(MAX_DETECTION_DIM / max(w, h), 1.0)
    if scale < 1.0:
        resized = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return resized, scale
    return image.copy(), 1.0


def detect_boxes(
    image_path: str,
    confidence: float = DEFAULT_CONFIDENCE,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> dict:
    """
    Detect board game boxes using OWLv2.

    Returns:
        {
            "detections": [{id, label, confidence, bbox: [x1,y1,x2,y2]}],
            "scale_factor": float,       # detection_res / original_res
            "detection_size": (w, h),     # resized image dimensions
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

    resized, scale_factor = resize_for_detection(original)
    det_w, det_h = resized.size
    print(f"[detect] {orig_w}x{orig_h} -> {det_w}x{det_h} (scale={scale_factor:.4f})")

    inputs = processor(text=[TEXT_QUERIES], images=resized, return_tensors="pt").to(device)

    print("[detect] Running inference...")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[det_h, det_w]], device=device)
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

    # NMS — highest confidence first
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    detections = _nms(detections, nms_iou)
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
