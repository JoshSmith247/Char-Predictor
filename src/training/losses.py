"""
Combined MSE + SSIM loss.

  loss = alpha * MSE(y_true, y_pred)  +  (1 - alpha) * (1 - SSIM(y_true, y_pred))

MSE drives pixel-accurate averaging; SSIM preserves structural/edge fidelity.
alpha defaults to 0.8 (configurable via config.yaml → training.loss_alpha).
"""
import tensorflow as tf


def combined_loss(alpha: float = 0.8):
    """Return a loss function with the given MSE weight alpha."""

    def loss_fn(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # tf.image.ssim expects [..., H, W, C] with values in [0, 1]
        ssim_val = tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim_loss = 1.0 - tf.reduce_mean(ssim_val)

        return alpha * mse + (1.0 - alpha) * ssim_loss

    loss_fn.__name__ = f"combined_loss_alpha{alpha}"
    return loss_fn
