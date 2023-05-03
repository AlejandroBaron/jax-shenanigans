from typing import Any

from ..weights_init import random_init
from ._Layer import Layer


class GCNLayer(Layer):
    def __init__(self, n_in: int, n_out: int, **kwargs) -> None:
        super().__init__(n_in=n_in, n_out=n_out, name="GCNLayer")
        self.W = random_init(shape=(n_in, n_out), **kwargs)

    def __call__(self, X, A) -> Any:
        return self.W(X @ A).T
