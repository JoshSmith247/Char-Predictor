"""
Preprocessing pipeline.

Converts a raw rendered or scraped character image into a normalized
64×64 float32 numpy array suitable for model input.

Pipeline steps:
  1. Load as grayscale
  2. Denoise (for JPEG sources)
  3. Binarize via Otsu threshold
  4. Tight-crop to glyph bounding box
  5. Pad to square with configurable margin
  6. Resize to target_size × target_size
  7. Normalize to [0, 1]  (ink ≈ 0, background ≈ 1)
"""
import numpy as np
import cv2
from pathlib import Path


class PreprocessingPipeline:
    def __init__(self, config: dict):
        preprocessing = config.get("preprocessing", {})
        self.target_size: int = preprocessing.get("target_size", 64)
        self.margin_ratio: float = preprocessing.get("glyph_margin_ratio", 0.1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_file(self, path: str | Path, denoise: bool = False) -> np.ndarray | None:
        """Load an image file and run the full pipeline.

        Returns a (target_size, target_size, 1) float32 array, or None if
        no glyph pixels are found (blank / missing glyph).
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return self.process_array(img, denoise=denoise)

    def process_array(self, img: np.ndarray, denoise: bool = False) -> np.ndarray | None:
        """Run the pipeline on a uint8 grayscale numpy array.

        Returns (target_size, target_size, 1) float32 or None if blank.
        """
        if denoise:
            img = cv2.fastNlMeansDenoising(img, h=10)

        # Binarize: ink=0, background=255
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Crop to glyph bounding box
        cropped = self._tight_crop(binary)
        if cropped is None:
            return None

        # Pad to square with margin
        padded = self._pad_to_square(cropped)

        # Resize
        resized = cv2.resize(
            padded,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        # Normalize to [0, 1] and add channel dim
        normalized = resized.astype(np.float32) / 255.0
        return normalized[:, :, np.newaxis]  # (H, W, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tight_crop(self, binary: np.ndarray) -> np.ndarray | None:
        """Crop the image tightly to the bounding box of the ink pixels."""
        # Ink pixels are 0; invert so they appear as non-zero for findNonZero
        inv = 255 - binary
        coords = cv2.findNonZero(inv)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        return binary[y : y + h, x : x + w]

    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        """Pad image to a square with a consistent margin around the glyph."""
        h, w = img.shape
        side = max(h, w)
        margin = max(1, int(side * self.margin_ratio))
        canvas_side = side + 2 * margin

        # White canvas (background value = 255)
        canvas = np.full((canvas_side, canvas_side), 255, dtype=np.uint8)

        # Center the glyph
        y_off = margin + (side - h) // 2
        x_off = margin + (side - w) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = img

        return canvas
