"""
scripts/vae.py

Variational Autoencoder that learns to generate images of a single character.

Usage (Python):
    from scripts.vae import CharVAE

    vae = CharVAE("A")
    vae.build_dataset()       # renders "A" in every downloaded font
    vae.build_model()
    vae.train(epochs=50)
    imgs = vae.generate(n=16) # (16, 64, 64) uint8 array
    vae.save_grid(imgs, "out.png")

Usage (CLI):
    python scripts/vae.py --target-char A --epochs 50
    python scripts/vae.py --target-char A --load --generate 16 --output out.png
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from font_downloader import FontDownloader

IMAGE_SIZE = 64
FONT_PT    = 48


# ---------------------------------------------------------------------------
# Reusable Keras pieces (module-level so load_model can find them)
# ---------------------------------------------------------------------------

class Sampling(tf.keras.layers.Layer):
    """Reparameterisation trick: z = mean + exp(0.5 * log_var) * epsilon."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


class _VAEModel(tf.keras.Model):
    """Wraps encoder + decoder; implements the combined VAE loss in train_step."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._loss_tracker  = tf.keras.metrics.Mean(name="loss")
        self._recon_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self._kl_tracker    = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self._loss_tracker, self._recon_tracker, self._kl_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)

            # Reconstruction: binary cross-entropy summed over pixels
            recon = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            # KL divergence from N(0,1)
            kl = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            loss = recon + kl

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._loss_tracker.update_state(loss)
        self._recon_tracker.update_state(recon)
        self._kl_tracker.update_state(kl)
        return {m.name: m.result() for m in self.metrics}


# ---------------------------------------------------------------------------
# CharVAE
# ---------------------------------------------------------------------------

class CharVAE:
    def __init__(
        self,
        target_char: str,
        fonts_dir: str = "fonts",
        model_dir: str = "model",
        latent_dim: int = 32,
    ):
        self.target_char = target_char
        self.fonts_dir   = fonts_dir
        self.model_dir   = model_dir
        self.latent_dim  = latent_dim
        self.encoder = None
        self.decoder = None
        self._vae    = None
        self._X: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 1. Font acquisition
    # ------------------------------------------------------------------

    def download_fonts(self, api_key: str, count: int = 100) -> None:
        """Download Google Fonts into self.fonts_dir."""
        FontDownloader(api_key=api_key).download(output_dir=self.fonts_dir, count=count)

    # ------------------------------------------------------------------
    # 2. Dataset  (only images of target_char)
    # ------------------------------------------------------------------

    def _collect_font_files(self) -> list[str]:
        paths = []
        for root, _, files in os.walk(self.fonts_dir):
            for fname in files:
                if fname.lower().endswith((".ttf", ".otf")):
                    paths.append(os.path.join(root, fname))
        return paths

    def _render_char(self, font_path: str) -> np.ndarray | None:
        try:
            pil_font = ImageFont.truetype(font_path, size=FONT_PT)
        except Exception:
            return None
        img  = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), self.target_char, font=pil_font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (IMAGE_SIZE - w) // 2 - bbox[0]
        y = (IMAGE_SIZE - h) // 2 - bbox[1]
        draw.text((x, y), self.target_char, fill=255, font=pil_font)
        return np.array(img, dtype=np.uint8)

    def build_dataset(self) -> np.ndarray:
        """
        Render target_char with every downloaded font.
        Returns float32 array of shape (N, IMAGE_SIZE, IMAGE_SIZE, 1).
        """
        font_files = self._collect_font_files()
        if not font_files:
            raise RuntimeError(
                f"No font files found in '{self.fonts_dir}'. Run download_fonts() first."
            )
        print(f"Rendering '{self.target_char}' from {len(font_files)} fonts...")
        images = [arr for f in font_files if (arr := self._render_char(f)) is not None]
        X = np.stack(images).astype(np.float32) / 255.0
        X = X[..., np.newaxis]
        self._X = X
        print(f"Dataset: {X.shape[0]} images of '{self.target_char}'.")
        return X

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        """
        Encoder  : 64×64×1  →  Conv×2  →  Dense(128)  →  [z_mean, z_log_var, z]
        Decoder  : latent    →  Dense   →  ConvTranspose×2  →  64×64×1
        """
        latent_dim = self.latent_dim

        # --- Encoder ---
        enc_in    = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
        x         = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(enc_in)
        x         = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
        x         = tf.keras.layers.Flatten()(x)
        x         = tf.keras.layers.Dense(128, activation="relu")(x)
        z_mean    = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z         = Sampling()([z_mean, z_log_var])
        self.encoder = tf.keras.Model(enc_in, [z_mean, z_log_var, z], name="encoder")

        # --- Decoder ---
        dec_in  = tf.keras.Input(shape=(latent_dim,))
        x       = tf.keras.layers.Dense(16 * 16 * 64, activation="relu")(dec_in)
        x       = tf.keras.layers.Reshape((16, 16, 64))(x)
        x       = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
        x       = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
        dec_out = tf.keras.layers.Conv2DTranspose(1,  3, padding="same", activation="sigmoid")(x)
        self.decoder = tf.keras.Model(dec_in, dec_out, name="decoder")

        self._vae = _VAEModel(self.encoder, self.decoder, name="char_vae")
        self._vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

        self.encoder.summary()
        self.decoder.summary()

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------

    def train(self, epochs: int = 50, batch_size: int = 32) -> None:
        """Train the VAE and save encoder + decoder to self.model_dir."""
        if self._vae is None:
            raise RuntimeError("Call build_model() first.")
        if self._X is None:
            raise RuntimeError("Call build_dataset() first.")

        self._vae.fit(self._X, epochs=epochs, batch_size=batch_size)
        self._save()

    def _save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        stem = f"vae_{self.target_char}"
        self.encoder.save(os.path.join(self.model_dir, f"{stem}_encoder.keras"))
        self.decoder.save(os.path.join(self.model_dir, f"{stem}_decoder.keras"))
        print(f"Saved encoder + decoder to '{self.model_dir}/'.")

    def load(self) -> None:
        """Load a previously trained VAE from self.model_dir."""
        stem    = f"vae_{self.target_char}"
        enc_path = os.path.join(self.model_dir, f"{stem}_encoder.keras")
        dec_path = os.path.join(self.model_dir, f"{stem}_decoder.keras")
        for p in (enc_path, dec_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"No saved model at '{p}'.")
        custom = {"Sampling": Sampling}
        self.encoder = tf.keras.models.load_model(enc_path, custom_objects=custom)
        self.decoder = tf.keras.models.load_model(dec_path, custom_objects=custom)

    # ------------------------------------------------------------------
    # 5. Generation
    # ------------------------------------------------------------------

    def generate(self, n: int = 1) -> np.ndarray:
        """
        Sample n images by drawing from the latent prior N(0, I).
        Returns uint8 array of shape (n, IMAGE_SIZE, IMAGE_SIZE).
        """
        if self.decoder is None:
            raise RuntimeError("Call build_model()+train() or load() first.")
        z    = np.random.normal(size=(n, self.latent_dim)).astype(np.float32)
        imgs = self.decoder.predict(z, verbose=0)   # (n, 64, 64, 1)
        return (imgs[..., 0] * 255).astype(np.uint8)

    def save_grid(self, images: np.ndarray, path: str, cols: int = 4) -> None:
        """
        Arrange images in a grid and write to a PNG file.
        images : uint8 array of shape (n, H, W)
        """
        n, h, w = images.shape
        rows = (n + cols - 1) // cols
        grid = Image.new("L", (cols * w, rows * h), color=0)
        for i, img in enumerate(images):
            grid.paste(Image.fromarray(img), ((i % cols) * w, (i // cols) * h))
        grid.save(path)
        print(f"Saved {n}-image grid to '{path}'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a VAE to generate character images.")
    parser.add_argument("--target-char", required=True, help="Character to learn (e.g. A)")
    parser.add_argument("--api-key",    default=None,        help="Google Fonts API key; omit if fonts already downloaded")
    parser.add_argument("--fonts-dir",  default="fonts")
    parser.add_argument("--model-dir",  default="model")
    parser.add_argument("--count",      type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--load",       action="store_true", help="Skip training; load saved model instead")
    parser.add_argument("--generate",   type=int, default=16, metavar="N", help="Number of images to generate")
    parser.add_argument("--output",     default="generated.png")
    args = parser.parse_args()

    vae = CharVAE(
        target_char=args.target_char,
        fonts_dir=args.fonts_dir,
        model_dir=args.model_dir,
        latent_dim=args.latent_dim,
    )

    if args.load:
        vae.load()
    else:
        if args.api_key:
            vae.download_fonts(api_key=args.api_key, count=args.count)
        vae.build_dataset()
        vae.build_model()
        vae.train(epochs=args.epochs, batch_size=args.batch_size)

    imgs = vae.generate(n=args.generate)
    vae.save_grid(imgs, args.output)
