"""
Font-level train/val/test splitting.

Splits on *font names*, not individual images, so the model is evaluated on
entirely unseen font styles. Writes one text file per split to data/splits/.
"""
import random
from pathlib import Path


def create_splits(
    processed_dir: str | Path,
    splits_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Discover all font names in processed_dir and split them.

    Font names are inferred from the filenames inside any character subdirectory.
    Returns {'train': [...], 'val': [...], 'test': [...]} and writes txt files.
    """
    processed_dir = Path(processed_dir)
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Collect font names from the first character directory found
    font_names: set[str] = set()
    for char_dir in processed_dir.iterdir():
        if not char_dir.is_dir() or char_dir.name.startswith("_"):
            continue
        for npy in char_dir.glob("*.npy"):
            if npy.stem != "mean_target":
                font_names.add(npy.stem)
        if font_names:
            break  # One directory is enough to discover all font names

    if not font_names:
        raise RuntimeError(f"No processed .npy files found in {processed_dir}")

    fonts = sorted(font_names)
    rng = random.Random(seed)
    rng.shuffle(fonts)

    n = len(fonts)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": fonts[:n_train],
        "val": fonts[n_train : n_train + n_val],
        "test": fonts[n_train + n_val :],
    }

    for split_name, font_list in splits.items():
        path = splits_dir / f"{split_name}_fonts.txt"
        path.write_text("\n".join(font_list))
        print(f"  {split_name}: {len(font_list)} fonts → {path}")

    return splits


def load_split(splits_dir: str | Path, split_name: str) -> list[str]:
    path = Path(splits_dir) / f"{split_name}_fonts.txt"
    return path.read_text().splitlines()
