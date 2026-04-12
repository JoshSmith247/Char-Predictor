"""Tests for the tf.data dataset pipeline."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import make_dataset
from src.data.splits import create_splits


def _build_fake_processed_dir(tmp_path: Path, n_fonts: int = 20, chars: str = "Aa0") -> Path:
    """Create a minimal processed/ directory structure for testing."""
    processed = tmp_path / "processed"
    font_names = [f"font_{i:03d}" for i in range(n_fonts)]

    for char in chars:
        char_dir = processed / char
        char_dir.mkdir(parents=True)
        arrays = []
        for name in font_names:
            arr = np.random.rand(64, 64, 1).astype(np.float32)
            np.save(str(char_dir / f"{name}.npy"), arr)
            arrays.append(arr)
        # Save mean target
        mean = np.mean(arrays, axis=0)
        np.save(str(char_dir / "mean_target.npy"), mean)

    return processed


class TestDataset:
    def test_output_shapes(self, tmp_path):
        processed = _build_fake_processed_dir(tmp_path, n_fonts=16, chars="Aa")
        font_list = [f"font_{i:03d}" for i in range(16)]
        ds = make_dataset(processed, font_list, K=4, batch_size=2, target_size=64, shuffle=False)
        inputs, target = next(iter(ds))
        assert inputs.shape == (2, 4, 64, 64, 1), f"Got {inputs.shape}"
        assert target.shape == (2, 64, 64, 1), f"Got {target.shape}"

    def test_value_range(self, tmp_path):
        processed = _build_fake_processed_dir(tmp_path, n_fonts=10, chars="A")
        font_list = [f"font_{i:03d}" for i in range(10)]
        ds = make_dataset(processed, font_list, K=3, batch_size=1, target_size=64)
        inputs, target = next(iter(ds))
        assert float(inputs.numpy().min()) >= 0.0
        assert float(inputs.numpy().max()) <= 1.5  # augmentation may add small delta

    def test_insufficient_fonts_raises(self, tmp_path):
        processed = _build_fake_processed_dir(tmp_path, n_fonts=3, chars="A")
        font_list = [f"font_{i:03d}" for i in range(3)]
        with pytest.raises(RuntimeError):
            make_dataset(processed, font_list, K=10, batch_size=1, target_size=64)

    def test_splits_disjoint(self, tmp_path):
        processed = _build_fake_processed_dir(tmp_path, n_fonts=30, chars="A")
        splits_dir = tmp_path / "splits"
        splits = create_splits(processed, splits_dir, train_ratio=0.8, val_ratio=0.1)
        all_sets = [set(splits["train"]), set(splits["val"]), set(splits["test"])]
        # No font should appear in two splits
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = all_sets[i] & all_sets[j]
                assert not overlap, f"Overlap between splits: {overlap}"

    def test_splits_cover_all_fonts(self, tmp_path):
        processed = _build_fake_processed_dir(tmp_path, n_fonts=30, chars="A")
        splits_dir = tmp_path / "splits"
        splits = create_splits(processed, splits_dir)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 30
