"""
tf.data pipeline for the Char-Predictor.

Each dataset example is:
  inputs : (K, target_size, target_size, 1)  float32 — K font images of one char
  target : (target_size, target_size, 1)     float32 — pixel-mean composite for that char

Training uses a fixed K (e.g. 8). The model's MeanAggregator is K-invariant so
inference can accept any number of images; the fixed-K training generalises.
"""
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


def _load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def _augment(img: np.ndarray) -> np.ndarray:
    """Apply mild per-image augmentation. img shape: (H, W, 1), float32 in [0,1]."""
    # Random brightness jitter ±0.1
    delta = np.random.uniform(-0.1, 0.1)
    img = np.clip(img + delta, 0.0, 1.0)
    return img


def _make_generator(
    char_font_map: dict[str, list[Path]],
    target_size: int,
    K: int,
    augment: bool,
    seed: int | None,
):
    """Return a generator function yielding (inputs, target) pairs."""
    chars = [c for c, paths in char_font_map.items() if len(paths) >= K]
    rng = random.Random(seed)

    def generator():
        while True:
            char = rng.choice(chars)
            font_paths = char_font_map[char]
            sampled = rng.sample(font_paths, K)

            # Load and optionally augment each image
            imgs = []
            for p in sampled:
                arr = _load_npy(str(p))
                if augment:
                    arr = _augment(arr)
                imgs.append(arr)
            inputs = np.stack(imgs, axis=0)  # (K, H, W, 1)

            # Load precomputed mean target
            mean_path = p.parent / "mean_target.npy"
            if not mean_path.exists():
                continue
            target = _load_npy(str(mean_path))  # (H, W, 1)

            yield inputs, target

    return generator


def make_dataset(
    processed_dir: str | Path,
    font_list: list[str],
    K: int,
    batch_size: int,
    target_size: int = 64,
    shuffle: bool = True,
    augment: bool = False,
    seed: int | None = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset for the given font split.

    Args:
        processed_dir: Root of preprocessed .npy files.
        font_list:     Fonts allowed for this split (train/val/test).
        K:             Number of fonts per input set (fixed during training).
        batch_size:    Examples per batch.
        target_size:   Spatial size of images (default 64).
        shuffle:       Whether to shuffle.
        augment:       Whether to apply per-image augmentation.
        seed:          Random seed for reproducibility.
    """
    processed_dir = Path(processed_dir)

    # Build {char: [path, ...]} index filtered to font_list
    font_set = set(font_list)
    char_font_map: dict[str, list[Path]] = {}

    for char_dir in sorted(processed_dir.iterdir()):
        if not char_dir.is_dir() or char_dir.name.startswith("_"):
            continue
        paths = [
            char_dir / f"{font}.npy"
            for font in font_set
            if (char_dir / f"{font}.npy").exists()
        ]
        if len(paths) >= K:
            char_font_map[char_dir.name] = paths

    if not char_font_map:
        raise RuntimeError(
            f"No character directories with >= {K} fonts found in {processed_dir}. "
            "Run scripts/preprocess.py and scripts/build_dataset.py first."
        )

    gen_fn = _make_generator(char_font_map, target_size, K, augment, seed)

    output_signature = (
        tf.TensorSpec(shape=(K, target_size, target_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(target_size, target_size, 1), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen_fn, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(buffer_size=512, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
