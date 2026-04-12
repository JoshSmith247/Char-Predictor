"""Tests for the CharPredictor model architecture."""
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.aggregator import MeanAggregator
from src.model.char_predictor import CharPredictor
from src.model.decoder import build_decoder
from src.model.encoder import build_encoder


class TestEncoder:
    def test_output_shape(self):
        encoder = build_encoder(target_size=64, latent_dim=256)
        x = tf.random.normal([4, 64, 64, 1])
        out = encoder(x)
        assert out.shape == (4, 256)

    def test_output_is_deterministic_in_inference_mode(self):
        encoder = build_encoder(target_size=64, latent_dim=256)
        x = tf.random.normal([2, 64, 64, 1])
        out1 = encoder(x, training=False)
        out2 = encoder(x, training=False)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-5)


class TestMeanAggregator:
    def test_output_shape(self):
        agg = MeanAggregator()
        x = tf.random.normal([3, 8, 256])
        out = agg(x)
        assert out.shape == (3, 256)

    def test_permutation_invariant(self):
        agg = MeanAggregator()
        x = tf.random.normal([1, 5, 64])
        shuffled = tf.gather(x, tf.random.shuffle(tf.range(5)), axis=1)
        out1 = agg(x)
        out2 = agg(shuffled)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-6)

    def test_single_input(self):
        agg = MeanAggregator()
        x = tf.random.normal([2, 1, 128])
        out = agg(x)
        assert out.shape == (2, 128)
        # Mean of one vector should equal that vector
        np.testing.assert_allclose(out.numpy(), x.numpy()[:, 0, :], atol=1e-6)


class TestDecoder:
    def test_output_shape(self):
        decoder = build_decoder(target_size=64, latent_dim=256)
        x = tf.random.normal([2, 256])
        out = decoder(x)
        assert out.shape == (2, 64, 64, 1)

    def test_output_value_range(self):
        decoder = build_decoder(target_size=64, latent_dim=256)
        x = tf.random.normal([4, 256])
        out = decoder(x)
        assert float(out.numpy().min()) >= 0.0
        assert float(out.numpy().max()) <= 1.0


class TestCharPredictor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = CharPredictor(target_size=64, latent_dim=256)

    def test_standard_output_shape(self):
        x = tf.random.normal([2, 8, 64, 64, 1])
        out = self.model(x, training=False)
        assert out.shape == (2, 64, 64, 1)

    def test_output_value_range(self):
        x = tf.random.normal([2, 8, 64, 64, 1])
        out = self.model(x, training=False)
        assert float(out.numpy().min()) >= 0.0
        assert float(out.numpy().max()) <= 1.0

    @pytest.mark.parametrize("K", [1, 3, 8, 20, 50])
    def test_variable_K(self, K):
        """Model must handle any number of input images at inference time."""
        x = tf.random.normal([1, K, 64, 64, 1])
        out = self.model(x, training=False)
        assert out.shape == (1, 64, 64, 1), f"Failed for K={K}"

    def test_k_invariance_of_identical_inputs(self):
        """Feeding K copies of the same image should give the same result as K=1."""
        single = tf.random.normal([1, 1, 64, 64, 1])
        repeated = tf.repeat(single, repeats=5, axis=1)  # (1, 5, 64, 64, 1)
        out1 = self.model(single, training=False)
        out5 = self.model(repeated, training=False)
        np.testing.assert_allclose(out1.numpy(), out5.numpy(), atol=1e-5)
