import jax.numpy as jnp

from .._Layer import Layer


class Sigmoid(Layer):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return 1 / (1 + jnp.exp(-x))
