"""
Training helper: builds callbacks and compiles the model.
"""
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from .losses import combined_loss


def build_callbacks(config: dict) -> list:
    training = config.get("training", {})
    checkpoint_dir = Path(training.get("checkpoint_dir", "checkpoints"))
    log_dir = Path(training.get("log_dir", "logs"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training.get("early_stopping_patience", 10),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=training.get("lr_reduce_factor", 0.5),
            patience=training.get("lr_reduce_patience", 5),
            min_lr=training.get("min_lr", 1e-6),
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            write_graph=False,
        ),
    ]


def compile_model(model: keras.Model, config: dict) -> None:
    training = config.get("training", {})
    lr = training.get("learning_rate", 1e-3)
    alpha = training.get("loss_alpha", 0.8)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=combined_loss(alpha=alpha),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
