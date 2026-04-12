"""Entry point: compute mean targets and create train/val/test splits."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.data.splits import create_splits
from src.utils.config import load_config


def compute_mean_targets(processed_dir: Path) -> None:
    """For each character directory, compute and save the pixel-wise mean
    of all font .npy files as mean_target.npy."""
    char_dirs = [d for d in sorted(processed_dir.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    print(f"Computing mean targets for {len(char_dirs)} characters...")

    for char_dir in tqdm(char_dirs):
        font_npys = [p for p in char_dir.glob("*.npy") if p.stem != "mean_target"]
        if not font_npys:
            continue

        arrays = [np.load(str(p)).astype(np.float32) for p in font_npys]
        mean_img = np.mean(arrays, axis=0)  # (H, W, 1)
        np.save(str(char_dir / "mean_target.npy"), mean_img)


def main():
    parser = argparse.ArgumentParser(description="Build mean targets and dataset splits.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])

    compute_mean_targets(processed_dir)

    print("Creating train/val/test splits...")
    create_splits(processed_dir, splits_dir)
    print("Done.")


if __name__ == "__main__":
    main()
