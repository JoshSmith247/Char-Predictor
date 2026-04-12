"""
Shared CNN encoder.

Takes a single (target_size, target_size, 1) image and produces a
flat (latent_dim,) feature vector. The same encoder weights are applied
to every image in the input set.

Architecture:
  4 × [Conv2D + BatchNorm + ReLU + MaxPool2D] — filters [32, 64, 128, 256]
  → Flatten
  → Dense(512) + BatchNorm + ReLU
  → Dense(latent_dim)
"""
import tensorflow as tf
from tensorflow import keras


def build_encoder(target_size: int = 64, latent_dim: int = 256) -> keras.Model:
    filters = [32, 64, 128, 256]
    inp = keras.Input(shape=(target_size, target_size, 1), name="encoder_input")
    x = inp

    for i, f in enumerate(filters):
        x = keras.layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"enc_conv{i+1}")(x)
        x = keras.layers.BatchNormalization(name=f"enc_bn{i+1}")(x)
        x = keras.layers.ReLU(name=f"enc_relu{i+1}")(x)
        x = keras.layers.MaxPool2D(2, name=f"enc_pool{i+1}")(x)

    x = keras.layers.Flatten(name="enc_flatten")(x)
    x = keras.layers.Dense(512, use_bias=False, name="enc_dense1")(x)
    x = keras.layers.BatchNormalization(name="enc_bn_dense")(x)
    x = keras.layers.ReLU(name="enc_relu_dense")(x)
    x = keras.layers.Dense(latent_dim, name="enc_output")(x)

    return keras.Model(inp, x, name="encoder")
