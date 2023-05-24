import jax.numpy as jnp
from jax import grad

from ..layers import Layer
from ._Model import Model


class Sequential(Model):
    def __init__(self, layers: list[Layer]) -> None:
        super().__init__()

        self.layers = layers

    def forward(self, *args, **kwargs):
        # why not jax.lax.reduce? because you need intermediate gradients
        output = self.layers[0](*args, **kwargs)
        for layer in self.layers[1:]:
            output = layer(output)
        return output
