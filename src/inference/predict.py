"""
Inference API for CharPredictor.

Usage:
    predictor = CharPredictorInference("models/final_model.keras", "config/config.yaml")
    composite = predictor.predict_from_paths(["font1.png", "font2.png", "font3.png"])
    predictor.save_output(composite, "output/A_composite.png")
"""
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.model.aggregator import MeanAggregator
from src.model.char_predictor import CharPredictor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.training.losses import combined_loss
from src.utils.config import load_config


class CharPredictorInference:
    def __init__(self, model_path: str, config_path: str = "config/config.yaml"):
        self.cfg = load_config(config_path)
        self.target_size: int = self.cfg["preprocessing"]["target_size"]
        self.preprocessor = PreprocessingPipeline(self.cfg)

        alpha = self.cfg["training"]["loss_alpha"]
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "CharPredictor": CharPredictor,
                "MeanAggregator": MeanAggregator,
                "loss_fn": combined_loss(alpha=alpha),
            },
        )

    def predict_from_paths(self, image_paths: list[str]) -> np.ndarray:
        """Accept N paths to raw character images and return a (H, W) composite array."""
        arrays = []
        for p in image_paths:
            arr = self.preprocessor.process_file(p)
            if arr is None:
                raise ValueError(f"Could not process image: {p}")
            arrays.append(arr)
        return self._run_model(arrays)

    def predict_from_arrays(self, images: list[np.ndarray]) -> np.ndarray:
        """Accept a list of preprocessed (H, W, 1) float32 arrays."""
        return self._run_model(images)

    def _run_model(self, arrays: list[np.ndarray]) -> np.ndarray:
        stacked = np.stack(arrays, axis=0)                  # (N, H, W, 1)
        inp = tf.expand_dims(stacked, axis=0)               # (1, N, H, W, 1)
        out = self.model(inp, training=False)               # (1, H, W, 1)
        return out.numpy()[0, :, :, 0]                      # (H, W)

    def save_output(self, composite: np.ndarray, output_path: str) -> None:
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        img = (composite * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(output_path, img)
        print(f"Saved composite to {output_path}")
