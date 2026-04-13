"""
scripts/train.py

Renders characters from downloaded fonts, builds a dataset, and trains a CNN
to predict which character is shown in an image.

Usage:
    from scripts.train import CharPredictor

    predictor = CharPredictor()
    predictor.download_fonts(api_key="YOUR_KEY", count=100)
    predictor.build_dataset()
    predictor.build_model()
    predictor.train(epochs=20)
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from font_downloader import FontDownloader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters the model will learn to classify
CHARACTERS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)
NUM_CLASSES = len(CHARACTERS)          # 62
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR = {i: ch for i, ch in enumerate(CHARACTERS)}

IMAGE_SIZE = 64   # pixels (square, grayscale)
FONT_PT    = 48   # point size used when rendering


# ---------------------------------------------------------------------------
# CharPredictor
# ---------------------------------------------------------------------------

class CharPredictor:
    def __init__(self, fonts_dir: str = "fonts", model_dir: str = "model"):
        self.fonts_dir = fonts_dir
        self.model_dir = model_dir
        self.model = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 1. Font acquisition
    # ------------------------------------------------------------------

    def download_fonts(self, api_key: str, count: int = 100) -> None:
        """Download `count` Google Fonts into self.fonts_dir."""
        downloader = FontDownloader(api_key=api_key)
        downloader.download(output_dir=self.fonts_dir, count=count)

    # ------------------------------------------------------------------
    # 2. Dataset construction
    # ------------------------------------------------------------------

    def _collect_font_files(self) -> list[str]:
        """Return all .ttf / .otf paths found under self.fonts_dir."""
        paths = []
        for root, _, files in os.walk(self.fonts_dir):
            for fname in files:
                if fname.lower().endswith((".ttf", ".otf")):
                    paths.append(os.path.join(root, fname))
        return paths

    def _render_char(self, char: str, font_path: str) -> np.ndarray | None:
        """
        Render a single character with the given font.
        Returns a (IMAGE_SIZE, IMAGE_SIZE) uint8 array, or None if the font
        does not contain a glyph for this character.
        """
        try:
            pil_font = ImageFont.truetype(font_path, size=FONT_PT)
        except Exception:
            return None

        img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        draw = ImageDraw.Draw(img)

        # Centre the glyph
        bbox = draw.textbbox((0, 0), char, font=pil_font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (IMAGE_SIZE - w) // 2 - bbox[0]
        y = (IMAGE_SIZE - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=255, font=pil_font)

        return np.array(img, dtype=np.uint8)

    def build_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Render every character in CHARACTERS with every downloaded font.

        Returns
        -------
        X : float32 array of shape (N, IMAGE_SIZE, IMAGE_SIZE, 1), values in [0, 1]
        y : int32  array of shape (N,), class indices into CHARACTERS
        """
        font_files = self._collect_font_files()
        if not font_files:
            raise RuntimeError(
                f"No font files found in '{self.fonts_dir}'. "
                "Run download_fonts() first."
            )

        print(f"Building dataset from {len(font_files)} font files...")

        images, labels = [], []
        for font_idx, font_path in enumerate(font_files, start=1):
            for char in CHARACTERS:
                arr = self._render_char(char, font_path)
                if arr is None:
                    continue
                images.append(arr)
                labels.append(CHAR_TO_IDX[char])

            if font_idx % 10 == 0 or font_idx == len(font_files):
                print(f"  Processed {font_idx}/{len(font_files)} fonts "
                      f"({len(images)} samples so far)")

        X = np.stack(images, axis=0).astype(np.float32) / 255.0
        X = X[..., np.newaxis]                          # add channel dim
        y = np.array(labels, dtype=np.int32)

        self._X, self._y = X, y
        print(f"Dataset ready: {X.shape[0]} samples, {NUM_CLASSES} classes.")
        return X, y

    # ------------------------------------------------------------------
    # 3. Model definition
    # ------------------------------------------------------------------

    def build_model(self) -> "tf.keras.Model":
        """
        Build and compile a small CNN suited for character recognition.

        Architecture
        ------------
        3 × (Conv2D → BatchNorm → MaxPool) blocks, then Dense head.
        """
        import tensorflow as tf

        inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))

        x = inputs
        for filters in (32, 64, 128):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs, name="char_predictor")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        self.model = model
        return model

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------

    def train(
        self,
        epochs: int = 20,
        batch_size: int = 64,
        val_split: float = 0.1,
    ) -> "tf.keras.callbacks.History":
        """
        Train the model on the prepared dataset and save it to self.model_dir.

        Requires build_dataset() and build_model() to have been called first
        (or pass X / y directly via the returned dataset tuple).
        """
        import tensorflow as tf

        if self.model is None:
            raise RuntimeError("Call build_model() before train().")
        if self._X is None or self._y is None:
            raise RuntimeError("Call build_dataset() before train().")

        os.makedirs(self.model_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.model_dir, "best.keras")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                save_best_only=True,
                monitor="val_accuracy",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        history = self.model.fit(
            self._X,
            self._y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            shuffle=True,
        )

        final_path = os.path.join(self.model_dir, "char_predictor.keras")
        self.model.save(final_path)
        print(f"Model saved to '{final_path}'.")

        return history

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the saved model from self.model_dir."""
        import tensorflow as tf

        path = os.path.join(self.model_dir, "char_predictor.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at '{path}'.")
        self.model = tf.keras.models.load_model(path)

    def predict(self, image: "np.ndarray | str") -> str:
        """
        Predict the character shown in an image.

        Parameters
        ----------
        image : str or numpy array
            File path to a grayscale PNG/JPG, or a pre-loaded uint8 array
            of shape (H, W) or (H, W, 1).

        Returns
        -------
        str : the predicted character
        """
        if self.model is None:
            raise RuntimeError("Call build_model()+train() or load_model() first.")

        if isinstance(image, str):
            arr = np.array(Image.open(image).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE)))
        else:
            arr = np.array(image)
            if arr.ndim == 3:
                arr = arr[..., 0]
            arr = np.array(Image.fromarray(arr).resize((IMAGE_SIZE, IMAGE_SIZE)))

        x = arr.astype(np.float32) / 255.0
        x = x[np.newaxis, ..., np.newaxis]              # (1, H, W, 1)

        probs = self.model.predict(x, verbose=0)[0]
        return IDX_TO_CHAR[int(np.argmax(probs))]


# ---------------------------------------------------------------------------
# Quick-start entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the char predictor.")
    parser.add_argument("--api-key", required=True, help="Google Fonts API key")
    parser.add_argument("--fonts-dir", default="fonts")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    predictor = CharPredictor(fonts_dir=args.fonts_dir, model_dir=args.model_dir)
    predictor.download_fonts(api_key=args.api_key, count=args.count)
    predictor.build_dataset()
    predictor.build_model()
    predictor.train(epochs=args.epochs)
