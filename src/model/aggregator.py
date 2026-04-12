"""
Mean aggregator layer.

Reduces a set of K feature vectors — (batch, K, latent_dim) — into a
single aggregated vector (batch, latent_dim) by taking the element-wise
mean across the K axis.

This layer has no trainable parameters and is permutation-invariant: the
output does not depend on the order of the K input images. It also handles
any K ≥ 1, so a model trained with K=8 still works at inference with K=3
or K=50.
"""
import tensorflow as tf
from tensorflow import keras


class MeanAggregator(keras.layers.Layer):
    def call(self, inputs):
        # inputs: (batch, K, latent_dim)
        return tf.reduce_mean(inputs, axis=1)  # (batch, latent_dim)

    def get_config(self):
        return super().get_config()
