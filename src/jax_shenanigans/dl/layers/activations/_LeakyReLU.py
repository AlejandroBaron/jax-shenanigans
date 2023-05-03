import jax.numpy as jnp

from .._Layer import Layer


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__(n_in=None, n_out=None, name="LeakyRELU")
        self.alpha = alpha
        assert self.alpha <= 0

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x < 0, self.alpha * x, x)  # jax.nn.leaky_relu
