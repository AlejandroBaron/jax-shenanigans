from typing import Any

from ..weights_init import random_init
from ._Layer import Layer


class GCNLayer(Layer):
    def __init__(self, n_out: int, n_in: int = None, **kwargs) -> None:
        super().__init__(n_in=n_in, n_out=n_out, name="GCNLayer")

        self.W = random_init(shape=(n_in, n_out), **kwargs)

    def _infer_n_in(self, X):
        self.n_in = self.n_in or len(X)

    def forward(self, X, A) -> Any:
        return self.W.T @ (A @ X)
