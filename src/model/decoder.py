"""
CNN decoder.

Takes the (latent_dim,) aggregated representation and upsamples to
(target_size, target_size, 1) using transposed convolutions.

Architecture (for target_size=32, latent_dim=128):
  Dense(2*2*256) + BatchNorm + ReLU → Reshape(2, 2, 256)
  4 × [Conv2DTranspose(F, 4, stride=2) + BatchNorm + ReLU]  — filters [256,128,64,32]
  Conv2D(1, 3, same) + Sigmoid  → (32, 32, 1)
"""
import math

import tensorflow as tf
from tensorflow import keras


def build_decoder(target_size: int = 64, latent_dim: int = 256) -> keras.Model:
    # Number of upsample stages needed to go from start_spatial to target_size.
    # Each stage doubles the spatial resolution (stride=2).
    n_stages = 4  # 2 → 4 → 8 → 16 → 32  (for target_size=32)
    start_spatial = target_size // (2 ** n_stages)  # 4 for target_size=64
    filters = [256, 128, 64, 32]

    inp = keras.Input(shape=(latent_dim,), name="decoder_input")
    x = keras.layers.Dense(
        start_spatial * start_spatial * 256,
        use_bias=False,
        name="dec_dense",
    )(inp)
    x = keras.layers.BatchNormalization(name="dec_bn_dense")(x)
    x = keras.layers.ReLU(name="dec_relu_dense")(x)
    x = keras.layers.Reshape((start_spatial, start_spatial, 256), name="dec_reshape")(x)

    for i, f in enumerate(filters):
        x = keras.layers.Conv2DTranspose(
            f, 4, strides=2, padding="same", use_bias=False, name=f"dec_up{i+1}"
        )(x)
        x = keras.layers.BatchNormalization(name=f"dec_bn{i+1}")(x)
        x = keras.layers.ReLU(name=f"dec_relu{i+1}")(x)

    x = keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="dec_output")(x)

    return keras.Model(inp, x, name="decoder")
