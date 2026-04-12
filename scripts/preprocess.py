"""Entry point: preprocess raw character images into .npy files."""
import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.config import load_config


def _process_one(args_tuple):
    png_path, npy_path, target_size, margin_ratio = args_tuple
    cfg = {"preprocessing": {"target_size": target_size, "glyph_margin_ratio": margin_ratio}}
    pipeline = PreprocessingPipeline(cfg)
    result = pipeline.process_file(png_path)
    if result is not None:
        np.save(str(npy_path), result)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw images to .npy files.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    target_size = cfg["preprocessing"]["target_size"]
    margin_ratio = cfg["preprocessing"]["glyph_margin_ratio"]

    # Collect all PNG files
    tasks = []
    for png_path in sorted(raw_dir.rglob("*.png")):
        # Mirror directory structure under processed/
        rel = png_path.relative_to(raw_dir)
        npy_path = processed_dir / rel.with_suffix(".npy")
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        if not npy_path.exists():
            tasks.append((png_path, npy_path, target_size, margin_ratio))

    print(f"Processing {len(tasks)} images with {args.workers} workers...")

    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(_process_one, tasks), total=len(tasks)))

    ok = sum(1 for r in results if r)
    skipped = len(results) - ok
    print(f"Done. {ok} saved, {skipped} skipped (blank/unreadable).")


if __name__ == "__main__":
    main()
