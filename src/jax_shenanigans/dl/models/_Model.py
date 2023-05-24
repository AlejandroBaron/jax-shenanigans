from typing import Any

from jax import jit


class Model:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return jit(self.forward)(*args, **kwargs)
