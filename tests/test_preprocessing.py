"""Tests for the preprocessing pipeline."""
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import PreprocessingPipeline

_CFG = {"preprocessing": {"target_size": 64, "glyph_margin_ratio": 0.1}}


def _make_synthetic_char_image(size=128) -> np.ndarray:
    """Create a synthetic grayscale image with a simple rectangle as a 'glyph'."""
    img = np.full((size, size), 255, dtype=np.uint8)
    # Draw a filled black rectangle in the centre
    margin = size // 4
    img[margin : size - margin, margin : size - margin] = 0
    return img


class TestPreprocessingPipeline:
    def setup_method(self):
        self.pipeline = PreprocessingPipeline(_CFG)

    def test_output_shape(self):
        img = _make_synthetic_char_image()
        result = self.pipeline.process_array(img)
        assert result is not None
        assert result.shape == (64, 64, 1)

    def test_output_dtype(self):
        img = _make_synthetic_char_image()
        result = self.pipeline.process_array(img)
        assert result.dtype == np.float32

    def test_output_value_range(self):
        img = _make_synthetic_char_image()
        result = self.pipeline.process_array(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_blank_image_returns_none(self):
        # All-white image has no ink pixels — should return None
        blank = np.full((64, 64), 255, dtype=np.uint8)
        result = self.pipeline.process_array(blank)
        assert result is None

    def test_different_target_sizes(self):
        for size in [32, 64, 128]:
            cfg = {"preprocessing": {"target_size": size, "glyph_margin_ratio": 0.1}}
            pipeline = PreprocessingPipeline(cfg)
            img = _make_synthetic_char_image()
            result = pipeline.process_array(img)
            assert result is not None
            assert result.shape == (size, size, 1)

    def test_glyph_is_centered(self):
        """After processing, the ink mass should be roughly centred."""
        img = _make_synthetic_char_image()
        result = self.pipeline.process_array(img)
        # Ink pixels have value near 0; find their centroid
        ink = (result[:, :, 0] < 0.5).astype(np.float32)
        if ink.sum() == 0:
            pytest.skip("No ink pixels in output")
        ys, xs = np.where(ink)
        cy, cx = ys.mean(), xs.mean()
        centre = 64 / 2
        assert abs(cy - centre) < 16, f"Glyph centroid y={cy:.1f} not near centre"
        assert abs(cx - centre) < 16, f"Glyph centroid x={cx:.1f} not near centre"
