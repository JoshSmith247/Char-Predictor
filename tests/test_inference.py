"""Tests for the inference API (without a real trained model)."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.pipeline import PreprocessingPipeline


def _make_synthetic_char_image(size=128) -> np.ndarray:
    img = np.full((size, size), 255, dtype=np.uint8)
    m = size // 4
    img[m : size - m, m : size - m] = 0
    return img


class TestPreprocessingInInferenceContext:
    """Verify the preprocessing step that inference relies on."""

    def test_process_array_output_shape(self):
        cfg = {"preprocessing": {"target_size": 64, "glyph_margin_ratio": 0.1}}
        pipeline = PreprocessingPipeline(cfg)
        img = _make_synthetic_char_image()
        result = pipeline.process_array(img)
        assert result is not None
        assert result.shape == (64, 64, 1)
        assert result.dtype == np.float32

    def test_process_file_missing_path_returns_none(self, tmp_path):
        cfg = {"preprocessing": {"target_size": 64, "glyph_margin_ratio": 0.1}}
        pipeline = PreprocessingPipeline(cfg)
        result = pipeline.process_file(str(tmp_path / "nonexistent.png"))
        assert result is None


class TestCharPredictorInferenceIntegration:
    """Integration test using a freshly-initialised (untrained) model."""

    def test_predict_from_arrays(self, tmp_path):
        import tensorflow as tf
        from src.model.char_predictor import CharPredictor
        from src.training.losses import combined_loss

        # Save a small untrained model to a temp path
        model = CharPredictor(target_size=64, latent_dim=64)
        dummy = tf.zeros([1, 2, 64, 64, 1])
        model(dummy)  # Build weights
        model_path = str(tmp_path / "test_model.keras")
        model.save(model_path)

        # Write a minimal config
        import yaml
        cfg = {
            "data": {"charset": "A"},
            "preprocessing": {"target_size": 64, "glyph_margin_ratio": 0.1},
            "training": {"loss_alpha": 0.8},
            "scraping": {},
            "inference": {"model_path": model_path, "output_dir": str(tmp_path)},
        }
        cfg_path = str(tmp_path / "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        from src.inference.predict import CharPredictorInference
        predictor = CharPredictorInference(model_path=model_path, config_path=cfg_path)

        # Two synthetic 64×64 arrays
        arrays = [np.random.rand(64, 64, 1).astype(np.float32) for _ in range(2)]
        composite = predictor.predict_from_arrays(arrays)

        assert composite.shape == (64, 64)
        assert composite.dtype == np.float32
        assert composite.min() >= 0.0
        assert composite.max() <= 1.0

    def test_save_output_creates_file(self, tmp_path):
        import tensorflow as tf
        import yaml
        from src.inference.predict import CharPredictorInference
        from src.model.char_predictor import CharPredictor

        model = CharPredictor(target_size=64, latent_dim=64)
        dummy = tf.zeros([1, 1, 64, 64, 1])
        model(dummy)
        model_path = str(tmp_path / "m.keras")
        model.save(model_path)

        cfg = {
            "data": {"charset": "A"},
            "preprocessing": {"target_size": 64, "glyph_margin_ratio": 0.1},
            "training": {"loss_alpha": 0.8},
            "scraping": {},
            "inference": {"model_path": model_path, "output_dir": str(tmp_path)},
        }
        cfg_path = str(tmp_path / "cfg.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        predictor = CharPredictorInference(model_path=model_path, config_path=cfg_path)
        composite = np.random.rand(64, 64).astype(np.float32)
        out_path = str(tmp_path / "out.png")
        predictor.save_output(composite, out_path)

        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0
