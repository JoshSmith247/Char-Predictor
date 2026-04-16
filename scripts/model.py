"""
scripts/model.py

Dual-encoder character style predictor.

Architecture:
    N images of char C (across many fonts)  →  Content Encoder  →  z_content
    K images of font F (various chars)      →  Style Encoder    →  z_style
                                                                        ↓
                                                              Concat + Dense(latent_dim)
                                                                        ↓
                                                                    Decoder  →  C in font F

Usage (Python):
    from scripts.model import CharStylePredictor

    p = CharStylePredictor(target_char="A")
    p.download_fonts(api_key="YOUR_KEY")
    p.build_dataset()
    p.build_model()
    p.train(epochs=50)
    img = p.predict(font_path="fonts/MyFont/MyFont-Regular.ttf")  # (64, 64) uint8
    p.save_grid([img], "predicted.png")

Usage (CLI):
    python scripts/model.py --target-char A --epochs 50
    python scripts/model.py --target-char A --load --font-path fonts/MyFont.ttf --output out.png
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
UPPERCASE  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# Keras building blocks
# ---------------------------------------------------------------------------

class _SetEncoder(tf.keras.layers.Layer):
    """
    Applies a shared single-image encoder to every image in a set,
    then mean-aggregates the embeddings (K-invariant, no learnable params here).

    Input  shape: (batch, K, H, W, C)
    Output shape: (batch, latent_dim)
    """
    def __init__(self, image_encoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.image_encoder = image_encoder

    def call(self, x, training=False):
        batch  = tf.shape(x)[0]
        k      = tf.shape(x)[1]
        x_flat = tf.reshape(x, (batch * k, IMAGE_SIZE, IMAGE_SIZE, 1))
        z_flat = self.image_encoder(x_flat, training=training)   # (batch*K, latent_dim)
        z      = tf.reshape(z_flat, (batch, k, -1))
        return tf.reduce_mean(z, axis=1)                          # (batch, latent_dim)


class _DualEncoderModel(tf.keras.Model):
    """
    Assembles content encoder, style encoder, fusion layer, and decoder.
    Implements a foreground-weighted binary cross-entropy loss in train_step.
    """
    def __init__(
        self,
        content_set_enc:  _SetEncoder,
        style_set_enc:    _SetEncoder,
        fusion:           tf.keras.layers.Layer,
        decoder:          tf.keras.Model,
        foreground_weight: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.content_set_enc   = content_set_enc
        self.style_set_enc     = style_set_enc
        self.fusion            = fusion
        self.decoder           = decoder
        self.foreground_weight = foreground_weight
        self._loss_tracker     = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self._loss_tracker]

    def call(self, inputs, training=False):
        content_imgs, style_imgs = inputs
        z_content = self.content_set_enc(content_imgs, training=training)
        z_style   = self.style_set_enc(style_imgs,     training=training)
        z_fused   = self.fusion(tf.concat([z_content, z_style], axis=-1), training=training)
        return self.decoder(z_fused, training=training)

    def train_step(self, data):
        (content_imgs, style_imgs), target = data
        with tf.GradientTape() as tape:
            pred      = self((content_imgs, style_imgs), training=True)
            pixel_bce = tf.keras.losses.binary_crossentropy(target, pred)
            weights   = 1.0 + tf.squeeze(target, -1) * (self.foreground_weight - 1.0)
            loss      = tf.reduce_mean(tf.reduce_sum(weights * pixel_bce, axis=(1, 2)))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self._loss_tracker.update_state(loss)
        return {"loss": self._loss_tracker.result()}


# ---------------------------------------------------------------------------
# CharStylePredictor
# ---------------------------------------------------------------------------

class CharStylePredictor:
    def __init__(
        self,
        target_char:  str,
        style_chars:  list[str] | None = None,
        fonts_dir:    str = "fonts",
        model_dir:    str = "model",
        latent_dim:   int = 16,
        n_content:    int = 8,
        k_style:      int = 4,
    ):
        """
        Parameters
        ----------
        target_char  : Single character to learn to predict (e.g. "A").
        style_chars  : Pool of characters used as style references.
                       K chars are sampled from this pool per training example.
                       Defaults to uppercase A-Z minus target_char.
        fonts_dir    : Directory containing downloaded .ttf / .otf files.
        model_dir    : Where to save / load model weights.
        latent_dim   : Embedding size for each encoder.
                       The fused vector is 2×latent_dim before the projection layer.
        n_content    : Number of content reference images per training example.
        k_style      : Number of style reference images per training example.
        """
        if len(target_char) != 1:
            raise ValueError("target_char must be a single character.")
        if style_chars is not None and len(style_chars) < k_style:
            raise ValueError(
                f"style_chars pool has {len(style_chars)} entries but k_style={k_style}."
            )

        self.target_char = target_char
        self.style_chars = style_chars if style_chars is not None else [
            c for c in UPPERCASE if c != target_char
        ]
        self.fonts_dir  = fonts_dir
        self.model_dir  = model_dir
        self.latent_dim = latent_dim
        self.n_content  = n_content
        self.k_style    = k_style

        self._model:      _DualEncoderModel | None = None
        self._dataset:    tf.data.Dataset | None   = None
        self._renders:    dict[str, dict[str, np.ndarray]] = {}
        self._font_paths: list[str] = []

    # ------------------------------------------------------------------
    # 1. Font acquisition
    # ------------------------------------------------------------------

    def download_fonts(self, api_key: str, count: int = 100) -> None:
        """Download Google Fonts into self.fonts_dir."""
        FontDownloader(api_key=api_key).download(output_dir=self.fonts_dir, count=count)

    # ------------------------------------------------------------------
    # 2. Dataset
    # ------------------------------------------------------------------

    def _collect_font_files(self) -> list[str]:
        paths = []
        for root, _, files in os.walk(self.fonts_dir):
            for fname in files:
                if fname.lower().endswith((".ttf", ".otf")):
                    paths.append(os.path.join(root, fname))
        return paths

    def _render_char(self, char: str, font_path: str) -> np.ndarray | None:
        """Render char with the given font. Returns (64, 64) float32 in [0,1], or None."""
        try:
            pil_font = ImageFont.truetype(font_path, size=FONT_PT)
        except Exception:
            return None
        img  = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=pil_font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (IMAGE_SIZE - w) // 2 - bbox[0]
        y = (IMAGE_SIZE - h) // 2 - bbox[1]
        draw.text((x, y), char, fill=255, font=pil_font)
        return np.array(img, dtype=np.float32) / 255.0

    def build_dataset(self) -> tf.data.Dataset:
        """
        Pre-renders target_char + every style_char in every usable font, then
        assembles a tf.data.Dataset of ((content_imgs, style_imgs), target) triplets.

        content_imgs : (n_content, 64, 64, 1)  — target_char rendered in other fonts
        style_imgs   : (k_style,   64, 64, 1)  — style_chars rendered in the target font
        target       : (64, 64, 1)              — target_char rendered in the target font

        Style chars are randomly sampled from self.style_chars per training example,
        which makes the model robust to different reference sets at inference time.
        """
        chars_needed = set(self.style_chars) | {self.target_char}
        font_paths   = self._collect_font_files()
        if not font_paths:
            raise RuntimeError(f"No font files found in '{self.fonts_dir}'. Run download_fonts() first.")

        print(f"Rendering {len(chars_needed)} chars × {len(font_paths)} fonts...")
        renders: dict[str, dict[str, np.ndarray]] = {}
        for i, fp in enumerate(font_paths, 1):
            rendered: dict[str, np.ndarray] = {}
            for ch in chars_needed:
                arr = self._render_char(ch, fp)
                if arr is not None:
                    rendered[ch] = arr[..., np.newaxis]   # (64, 64, 1)
            if len(rendered) == len(chars_needed):        # skip fonts missing any needed glyph
                renders[fp] = rendered
            if i % 20 == 0 or i == len(font_paths):
                print(f"  {i}/{len(font_paths)} fonts scanned, {len(renders)} usable")

        usable = list(renders.keys())
        if len(usable) < self.n_content + 1:
            raise RuntimeError(
                f"Only {len(usable)} usable fonts; need at least {self.n_content + 1}."
            )

        self._renders    = renders
        self._font_paths = usable

        # Build arrays — one triplet per font, with randomly sampled style/content refs
        rng = np.random.default_rng(42)
        content_list, style_list, target_list = [], [], []

        for fp in usable:
            other_fonts = [f for f in usable if f != fp]
            if len(other_fonts) < self.n_content:
                continue
            content_fonts  = rng.choice(other_fonts, size=self.n_content, replace=False)
            sampled_styles = rng.choice(self.style_chars, size=self.k_style, replace=False)

            content_list.append(np.stack([renders[f][self.target_char] for f in content_fonts]))
            style_list.append(np.stack([renders[fp][sc] for sc in sampled_styles]))
            target_list.append(renders[fp][self.target_char])

        ds = tf.data.Dataset.from_tensor_slices((
            (np.stack(content_list).astype(np.float32),
             np.stack(style_list).astype(np.float32)),
            np.stack(target_list).astype(np.float32),
        ))
        self._dataset = ds
        print(f"Dataset: {len(target_list)} triplets from {len(usable)} fonts.")
        return ds

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------

    def _build_components(self) -> None:
        """
        Constructs all sub-models and assembles _DualEncoderModel.
        Called by both build_model() (first train) and load() (restoring weights).
        """
        latent_dim = self.latent_dim

        def _img_encoder(name: str) -> tf.keras.Model:
            """Single-image CNN: (64, 64, 1) → (latent_dim,)"""
            inp = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
            x   = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
            x   = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
            x   = tf.keras.layers.Flatten()(x)
            x   = tf.keras.layers.Dense(128, activation="relu")(x)
            z   = tf.keras.layers.Dense(latent_dim)(x)
            return tf.keras.Model(inp, z, name=name)

        # Separate weights for content vs style encoders
        content_img_enc = _img_encoder("content_img_enc")
        style_img_enc   = _img_encoder("style_img_enc")

        # Decoder: latent_dim → 64×64×1
        BN      = tf.keras.layers.BatchNormalization
        dec_in  = tf.keras.Input(shape=(latent_dim,))
        x       = tf.keras.layers.Dense(16 * 16 * 128, activation="relu")(dec_in)
        x       = tf.keras.layers.Reshape((16, 16, 128))(x)
        x       = tf.keras.layers.UpSampling2D(2)(x)                               # → 32×32
        x       = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x       = BN()(x)
        x       = tf.keras.layers.Dropout(0.1)(x)
        x       = tf.keras.layers.UpSampling2D(2)(x)                               # → 64×64
        x       = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x       = BN()(x)
        x       = tf.keras.layers.Dropout(0.1)(x)
        x       = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        dec_out = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
        decoder = tf.keras.Model(dec_in, dec_out, name="decoder")

        # Fusion: concat(z_content, z_style) → z_fused, projects 2×latent_dim → latent_dim
        fusion = tf.keras.layers.Dense(latent_dim, activation="relu", name="fusion")

        self._model = _DualEncoderModel(
            content_set_enc=_SetEncoder(content_img_enc, name="content_set_enc"),
            style_set_enc=_SetEncoder(style_img_enc,     name="style_set_enc"),
            fusion=fusion,
            decoder=decoder,
            name="char_style_predictor",
        )

        # Trace the model to materialise all weight variables before save/load
        dummy_c = tf.zeros((1, self.n_content, IMAGE_SIZE, IMAGE_SIZE, 1))
        dummy_s = tf.zeros((1, self.k_style,   IMAGE_SIZE, IMAGE_SIZE, 1))
        self._model((dummy_c, dummy_s), training=False)

    def build_model(self, lr: float = 3e-4, foreground_weight: float = 5.0) -> None:
        """
        Content Encoder (per image, mean-pooled over n_content):
            64×64×1 → Conv(32) → Conv(64) → Flatten → Dense(128) → Dense(latent_dim)

        Style Encoder (same structure, separate weights, mean-pooled over k_style):
            64×64×1 → Conv(32) → Conv(64) → Flatten → Dense(128) → Dense(latent_dim)

        Fusion:
            Concat(z_content, z_style)[2×latent_dim] → Dense(latent_dim, relu)

        Decoder:
            latent_dim → Dense(16×16×128) → Reshape →
            UpSample+Conv(128)+BN → UpSample+Conv(64)+BN → Conv(64) → Conv(1, sigmoid)
        """
        self._build_components()
        self._model.foreground_weight = foreground_weight
        self._model.compile(optimizer=tf.keras.optimizers.Adam(lr))
        self._model.summary()

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------

    def train(self, epochs: int = 50, batch_size: int = 32) -> None:
        if self._model is None:
            raise RuntimeError("Call build_model() first.")
        if self._dataset is None:
            raise RuntimeError("Call build_dataset() first.")
        ds = self._dataset.shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        self._model.fit(ds, epochs=epochs)
        self._save()

    def _weights_path(self) -> str:
        return os.path.join(self.model_dir, f"style_{self.target_char}.weights.h5")

    def _save(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        self._model.save_weights(self._weights_path())
        print(f"Saved weights to '{self._weights_path()}'.")

    def load(self) -> None:
        """Load previously saved weights. Rebuilds the model architecture first."""
        path = self._weights_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved weights at '{path}'.")
        self._build_components()
        self._model.load_weights(path)

    # ------------------------------------------------------------------
    # 5. Inference
    # ------------------------------------------------------------------

    def predict(self, font_path: str, style_chars: list[str] | None = None) -> np.ndarray:
        """
        Predict what target_char looks like in the font at font_path.

        Parameters
        ----------
        font_path   : Path to a .ttf / .otf file for the target font.
        style_chars : Characters to render from font_path as style references.
                      Defaults to the first k_style chars of self.style_chars.

        Returns
        -------
        uint8 array of shape (IMAGE_SIZE, IMAGE_SIZE)
        """
        if self._model is None:
            raise RuntimeError("Call build_model()+train() or load() first.")
        if not self._font_paths:
            raise RuntimeError("Call build_dataset() first (needed for content references).")

        # Style images: render k_style chars from the target font
        sc_pool = style_chars or self.style_chars
        style_imgs = []
        for sc in sc_pool[: self.k_style]:
            arr = self._render_char(sc, font_path)
            if arr is None:
                raise RuntimeError(f"Font '{font_path}' could not render '{sc}'.")
            style_imgs.append(arr[..., np.newaxis])
        style_arr = np.stack(style_imgs)[np.newaxis].astype(np.float32)
        # (1, k_style, 64, 64, 1)

        # Content images: sample n_content fonts from the training set
        rng     = np.random.default_rng()
        sampled = rng.choice(self._font_paths, size=self.n_content, replace=False)
        content_arr = np.stack(
            [self._renders[f][self.target_char] for f in sampled]
        )[np.newaxis].astype(np.float32)
        # (1, n_content, 64, 64, 1)

        pred = self._model((content_arr, style_arr), training=False).numpy()
        img  = pred[0, ..., 0]   # (64, 64)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)
        return (img * 255).astype(np.uint8)

    def save_grid(self, images: list[np.ndarray], path: str, cols: int = 4) -> None:
        """Arrange (64, 64) uint8 images in a grid and write to a PNG file."""
        arr  = np.stack(images)
        n, h, w = arr.shape
        rows = (n + cols - 1) // cols
        grid = Image.new("L", (cols * w, rows * h), color=0)
        for i, img in enumerate(arr):
            grid.paste(Image.fromarray(img), ((i % cols) * w, (i // cols) * h))
        grid.save(path)
        print(f"Saved {n}-image grid to '{path}'.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the dual-encoder char style predictor.")
    parser.add_argument("--target-char",       required=True,              help="Character to predict (e.g. A)")
    parser.add_argument("--api-key",           default=None,               help="Google Fonts API key; omit if fonts already downloaded")
    parser.add_argument("--fonts-dir",         default="fonts")
    parser.add_argument("--model-dir",         default="model")
    parser.add_argument("--count",             type=int,   default=100,    help="Number of fonts to download")
    parser.add_argument("--latent-dim",        type=int,   default=16,     help="Embedding size per encoder (default: 16)")
    parser.add_argument("--n-content",         type=int,   default=8,      help="Content reference images per example (default: 8)")
    parser.add_argument("--k-style",           type=int,   default=4,      help="Style reference images per example (default: 4)")
    parser.add_argument("--epochs",            type=int,   default=50)
    parser.add_argument("--batch-size",        type=int,   default=32)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--foreground-weight", type=float, default=5.0,    help="Loss weight for foreground pixels (default: 5.0)")
    parser.add_argument("--load",              action="store_true",         help="Skip training; load saved weights instead")
    parser.add_argument("--font-path",         default=None,               help="Font to run inference on after training/loading")
    parser.add_argument("--output",            default="predicted.png")
    args = parser.parse_args()

    predictor = CharStylePredictor(
        target_char=args.target_char,
        fonts_dir=args.fonts_dir,
        model_dir=args.model_dir,
        latent_dim=args.latent_dim,
        n_content=args.n_content,
        k_style=args.k_style,
    )

    # build_dataset() is always needed: content references are drawn from it at inference time
    if args.api_key:
        predictor.download_fonts(api_key=args.api_key, count=args.count)
    predictor.build_dataset()

    if args.load:
        predictor.load()
    else:
        predictor.build_model(lr=args.lr, foreground_weight=args.foreground_weight)
        predictor.train(epochs=args.epochs, batch_size=args.batch_size)

    if args.font_path:
        img = predictor.predict(font_path=args.font_path)
        predictor.save_grid([img], args.output, cols=1)
