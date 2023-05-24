from typing import Any

from jax import numpy as jnp

from ..weights_init import random_init
from ._Layer import Layer


class GCNLayer(Layer):
    def __init__(self, A: jnp.array, n_out: int, n_in: int = None, **kwargs) -> None:
        super().__init__(n_in=n_in, n_out=n_out, name="GCNLayer")

        self.W = random_init(shape=(n_in, n_out), **kwargs)
        self.A = A

    def _infer_n_in(self, X):
        self.n_in = self.n_in or len(X)

    def forward(self, X) -> Any:
        return (self.A @ X) @ self.W
