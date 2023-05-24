from typing import Any

from jax import jit


class Layer:
    def __init__(self, n_in: int = None, n_out: int = None, name: str = None) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.name = name

    @property
    def shape(self):
        return (self.n_in, self.n_out)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return jit(self.forward)(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name}. Shape: {self.shape}"

    def forward(self, X):
        raise NotImplementedError
