from typing import Any


class Layer:
    def __init__(self, n_in: int = None, n_out: int = None, name: str = None) -> None:
        self.n_in = n_in
        self.n_out = n_out
        self.name = name

    @property
    def shape(self):
        return (self.n_in, self.n_out)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self._name}. Shape: {self.shape}"
