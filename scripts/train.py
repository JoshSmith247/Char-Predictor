"""Entry point: train the CharPredictor model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import make_dataset
from src.data.splits import load_split
from src.model.char_predictor import CharPredictor
from src.training.trainer import build_callbacks, compile_model
from src.utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Train the CharPredictor model.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    p.add_argument("--steps-per-epoch", type=int, default=500,
                   help="Steps per epoch (generator is infinite; this caps each epoch)")
    p.add_argument("--val-steps", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    training_cfg = cfg["training"]
    processed_dir = cfg["data"]["processed_dir"]
    splits_dir = cfg["data"]["splits_dir"]
    K = training_cfg["K"]
    batch_size = training_cfg["batch_size"]
    target_size = cfg["preprocessing"]["target_size"]
    epochs = args.epochs or training_cfg["epochs"]

    print("Loading splits...")
    train_fonts = load_split(splits_dir, "train")
    val_fonts = load_split(splits_dir, "val")
    print(f"  train: {len(train_fonts)} fonts, val: {len(val_fonts)} fonts")

    print("Building datasets...")
    train_ds = make_dataset(
        processed_dir, train_fonts, K=K, batch_size=batch_size,
        target_size=target_size, shuffle=True, augment=True,
    )
    val_ds = make_dataset(
        processed_dir, val_fonts, K=K, batch_size=batch_size,
        target_size=target_size, shuffle=False, augment=False,
    )

    print("Building model...")
    model = CharPredictor(
        target_size=target_size,
        latent_dim=cfg["model"]["latent_dim"],
    )
    compile_model(model, cfg)

    # Warm up the model to print a summary
    import tensorflow as tf
    dummy = tf.zeros([1, K, target_size, target_size, 1])
    model(dummy)
    model.summary()

    print(f"Training for {epochs} epochs...")
    callbacks = build_callbacks(cfg)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.val_steps,
        callbacks=callbacks,
    )

    model_path = Path(cfg["inference"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
