"""
CharPredictor: full model.

Wires together:
  encoder    — shared CNN, applied independently to each of the K input images
  aggregator — permutation-invariant mean over the K feature vectors
  decoder    — CNN that generates the 64×64 composite image

Forward pass:
  inputs: (batch, K, H, W, 1)
  → flatten batch and K: (batch*K, H, W, 1)
  → encoder:             (batch*K, latent_dim)
  → reshape:             (batch, K, latent_dim)
  → mean aggregation:    (batch, latent_dim)
  → decoder:             (batch, H, W, 1)
"""
import tensorflow as tf
from tensorflow import keras

from .aggregator import MeanAggregator
from .decoder import build_decoder
from .encoder import build_encoder


class CharPredictor(keras.Model):
    def __init__(self, target_size: int = 64, latent_dim: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.latent_dim = latent_dim

        self.encoder = build_encoder(target_size=target_size, latent_dim=latent_dim)
        self.aggregator = MeanAggregator(name="mean_aggregator")
        self.decoder = build_decoder(target_size=target_size, latent_dim=latent_dim)

    def call(self, inputs, training=False):
        # inputs: (batch, K, H, W, 1)
        shape = tf.shape(inputs)
        batch_size = shape[0]
        K = shape[1]

        # Process all K images in one encoder pass
        flat = tf.reshape(inputs, [batch_size * K, self.target_size, self.target_size, 1])
        features = self.encoder(flat, training=training)             # (batch*K, latent_dim)
        features = tf.reshape(features, [batch_size, K, self.latent_dim])  # (batch, K, latent_dim)

        aggregated = self.aggregator(features)                        # (batch, latent_dim)
        output = self.decoder(aggregated, training=training)          # (batch, H, W, 1)
        return output

    def get_config(self):
        return {
            "target_size": self.target_size,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
