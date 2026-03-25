"""
Stages 3 + 4: High-res polygon crop, then tiered identification.

The crop step is CRITICAL: polygon coordinates from segment.py are in
detection resolution — we scale them back to original resolution, crop
from the full-res photo, and mask everything outside the polygon. This
gives the identification models the sharpest possible input.

Tiers (cheapest first):
  A. EasyOCR — extract text, match against game list
  B. CLIP    — visual matching against game name embeddings
  C. OpenRouter — VLM fallback for ambiguous cases (TODO)
"""

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Stage 3: High-res polygon crop
# ---------------------------------------------------------------------------

def polygon_crop(
    image_path: str,
    polygon: list[list[float]],
    scale_factor: float,
) -> np.ndarray:
    """
    Crop from the ORIGINAL high-res image using a polygon defined in detection resolution.

    1. Scale polygon coordinates: detection-res → original-res
    2. Get bounding rect of scaled polygon
    3. Crop that rect from the original image
    4. Mask out everything outside the polygon (black fill)

    Returns BGR numpy array of the masked crop.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    ih, iw = image.shape[:2]

    # Scale polygon back to original resolution
    # scale_factor = detection_res / original_res, so divide to go back
    scaled = np.array(polygon, dtype=np.float64) / scale_factor
    scaled = np.clip(scaled, 0, [iw - 1, ih - 1]).astype(np.int32)

    # Bounding rect of the polygon
    x, y, w, h = cv2.boundingRect(scaled)
    x = max(0, x)
    y = max(0, y)
    w = min(w, iw - x)
    h = min(h, ih - y)

    if w <= 0 or h <= 0:
        return np.zeros((64, 64, 3), dtype=np.uint8)

    # Crop the bounding rect from the original
    crop = image[y : y + h, x : x + w].copy()

    # Build mask in crop-local coordinates
    local_poly = scaled - np.array([x, y])
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_poly], 255)

    # Apply mask — black outside polygon
    return cv2.bitwise_and(crop, crop, mask=mask)


# ---------------------------------------------------------------------------
# Stage 4: Tiered identification
# ---------------------------------------------------------------------------

class GameIdentifier:
    """Identifies board games from polygon crops using tiered methods."""

    def __init__(self, game_names: list[str], game_images: dict[str, list] | None = None):
        """
        Args:
            game_names: List of game names from the user's BGG collection.
                        Used to constrain identification results.
            game_images: Optional {game_name: [list of image paths]} for CLIP
                         image-to-image matching. Multiple images per game
                         (front cover, 3D render, spine) improve matching.
        """
        self.game_names = game_names
        self.game_images = game_images or {}
        self._ocr_reader = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clip_text_features = None
        self._clip_image_features = None
        self._clip_image_names = None  # parallel list: game name per embedding row

    def identify(self, crop: np.ndarray) -> dict | None:
        """
        Run tiered identification on a polygon crop.

        Returns dict with:
            method: "ocr" | "clip" | None
            game_name: str
            confidence: float
            detail: dict (method-specific data)
        Or None if unidentified.
        """
        # Tier A: EasyOCR
        result = self._try_ocr(crop)
        if result and result["confidence"] >= 0.60:
            return result

        # Tier B: CLIP image-to-image matching (if box art images are cached)
        if self.game_images:
            result = self._try_clip_image(crop)
            if result and result["confidence"] >= 0.45:
                return result

        # Tier C: CLIP text-to-image matching
        result = self._try_clip(crop)
        if result and result["confidence"] >= 0.35:
            return result

        return None

    # -------------------------------------------------------------------
    # Tier A: EasyOCR
    # -------------------------------------------------------------------

    def _try_ocr(self, crop: np.ndarray) -> dict | None:
        """Extract text via OCR, match fragments against game names."""
        import torch

        if self._ocr_reader is None:
            import easyocr
            gpu = torch.cuda.is_available()
            self._ocr_reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

        results = self._ocr_reader.readtext(crop)
        if not results:
            return None

        # Collect text fragments with their confidences
        fragments = []
        for bbox, text, conf in results:
            text = text.strip()
            if len(text) >= 2:
                fragments.append({"text": text, "confidence": conf})

        if not fragments:
            return None

        # Concatenate fragments and try matching against game names
        full_text = " ".join(f["text"] for f in fragments)
        best_match, best_score = self._match_ocr_text(full_text)

        if best_match and best_score >= 0.60:
            return {
                "method": "ocr",
                "game_name": best_match,
                "confidence": round(best_score, 4),
                "detail": {"fragments": fragments, "full_text": full_text},
            }

        # Also try individual fragments — sometimes one fragment IS the game name
        for frag in sorted(fragments, key=lambda f: f["confidence"], reverse=True):
            match, score = self._match_ocr_text(frag["text"])
            if match and score >= 0.75:
                return {
                    "method": "ocr",
                    "game_name": match,
                    "confidence": round(score, 4),
                    "detail": {"fragments": fragments, "matched_fragment": frag["text"]},
                }

        return None

    def _match_ocr_text(self, text: str) -> tuple[str | None, float]:
        """Match OCR text against game names using embedding similarity."""
        if not self.game_names or not text.strip():
            return None, 0.0

        from sentence_transformers import SentenceTransformer

        # Lazy-load a lightweight model for text matching
        if not hasattr(self, "_text_model"):
            self._text_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._text_embeddings = self._text_model.encode(
                self.game_names, normalize_embeddings=True
            )

        query_emb = self._text_model.encode([text], normalize_embeddings=True)
        scores = (query_emb @ self._text_embeddings.T)[0]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= 0.50:
            return self.game_names[best_idx], best_score
        return None, 0.0

    # -------------------------------------------------------------------
    # Tier B: CLIP visual matching
    # -------------------------------------------------------------------

    def _try_clip(self, crop: np.ndarray) -> dict | None:
        """Match crop image against game names using CLIP zero-shot."""
        import torch

        if self._clip_model is None:
            self._init_clip()

        # Prepare image
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = self._clip_preprocess(pil_crop).unsqueeze(0)

        device = next(self._clip_model.parameters()).device
        image_input = image_input.to(device)

        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity against pre-computed text features
        similarities = (image_features @ self._clip_text_features.T).squeeze(0)
        scores = similarities.cpu().numpy()

        top_k = min(5, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        best_idx = top_indices[0]
        best_score = float(scores[best_idx])

        candidates = [
            {"name": self.game_names[idx], "score": round(float(scores[idx]), 4)}
            for idx in top_indices
        ]

        if best_score >= 0.35:
            return {
                "method": "clip",
                "game_name": self.game_names[best_idx],
                "confidence": round(best_score, 4),
                "detail": {"candidates": candidates},
            }
        return None

    def _init_clip(self):
        """Load CLIP model and pre-compute text embeddings for game names."""
        import torch
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[identify] Loading CLIP on {device}...")

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()

        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_tokenizer = tokenizer

        # Pre-compute text embeddings with a descriptive template
        prompts = [f"a photo of the board game {name}" for name in self.game_names]
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._clip_text_features = text_features
        print(f"[identify] CLIP ready — {len(self.game_names)} text embeddings cached")

        # Also pre-compute image embeddings if box art images are available
        if self.game_images:
            self._init_clip_images()

    def _init_clip_images(self):
        """Pre-compute CLIP image embeddings for cached box art images.

        Supports multiple images per game (front cover, 3D render, spine, etc.).
        Each image gets its own embedding row; _clip_image_names tracks which
        game name each row belongs to. _try_clip_image() groups by game name
        and takes the max score.
        """
        import torch

        device = next(self._clip_model.parameters()).device
        names = []
        tensors = []

        for name, img_paths in self.game_images.items():
            # img_paths is a list of Paths (one or more images per game)
            for img_path in img_paths:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    tensor = self._clip_preprocess(pil_img).unsqueeze(0)
                    tensors.append(tensor)
                    names.append(name)
                except Exception:
                    continue

        if not tensors:
            print("[identify] No valid box art images for CLIP image matching")
            return

        batch = torch.cat(tensors, dim=0).to(device)

        # Encode in batches to avoid OOM on large collections
        all_features = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(batch), batch_size):
                chunk = batch[i : i + batch_size]
                features = self._clip_model.encode_image(chunk)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features)

        self._clip_image_features = torch.cat(all_features, dim=0)
        self._clip_image_names = names

        unique_games = len(set(names))
        print(f"[identify] CLIP image embeddings: {len(names)} images for {unique_games} games")

    # -------------------------------------------------------------------
    # Tier B: CLIP image-to-image matching
    # -------------------------------------------------------------------

    def _try_clip_image(self, crop: np.ndarray) -> dict | None:
        """Match crop against cached box art images using CLIP image embeddings.

        With multiple reference images per game, we compute cosine similarity
        against ALL embeddings, then group by game name and take the max score
        per game. This way a 3D render or spine image that matches well will
        boost that game's score even if the front cover didn't match.
        """
        import torch
        from collections import defaultdict

        if self._clip_model is None:
            self._init_clip()

        if self._clip_image_features is None or len(self._clip_image_features) == 0:
            return None

        # Encode the crop
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = self._clip_preprocess(pil_crop).unsqueeze(0)

        device = next(self._clip_model.parameters()).device
        image_input = image_input.to(device)

        with torch.no_grad():
            image_features = self._clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity against ALL image embeddings
        similarities = (image_features @ self._clip_image_features.T).squeeze(0)
        scores = similarities.cpu().numpy()

        # Group scores by game name, take max per game
        game_max_scores: dict[str, float] = defaultdict(float)
        for idx, score in enumerate(scores):
            name = self._clip_image_names[idx]
            s = float(score)
            if s > game_max_scores[name]:
                game_max_scores[name] = s

        # Sort games by their max score
        sorted_games = sorted(game_max_scores.items(), key=lambda x: -x[1])
        top_k = min(5, len(sorted_games))
        top_games = sorted_games[:top_k]

        best_name, best_score = top_games[0]
        candidates = [
            {"name": name, "score": round(score, 4)}
            for name, score in top_games
        ]

        # Z-score check on per-game max scores
        all_max_scores = np.array(list(game_max_scores.values()))
        mean_score = float(all_max_scores.mean())
        std_score = float(all_max_scores.std())
        z_score = (best_score - mean_score) / std_score if std_score > 0 else 0.0

        if z_score < 3.0:
            return None

        # Margin check between top-1 and top-2 games
        if len(top_games) >= 2:
            second_score = top_games[1][1]
            margin = best_score - second_score
            if margin < 0.06:
                return None  # ambiguous — let Tier C try

        if best_score >= 0.45:
            return {
                "method": "clip_image",
                "game_name": best_name,
                "confidence": round(best_score, 4),
                "detail": {"candidates": candidates, "z_score": round(z_score, 2)},
            }
        return None


# ---------------------------------------------------------------------------
# Tier C: OpenRouter VLM fallback (TODO)
# ---------------------------------------------------------------------------

def identify_with_openrouter(crop: np.ndarray, candidates: list[str]) -> dict | None:
    """
    Send crop to a VLM via OpenRouter free tier for disambiguation.

    TODO: Implement when CLIP returns ambiguous top-k results.
    Would send the crop image + candidate list to e.g. Qwen2.5-VL
    and ask which game is shown.
    """
    return None
