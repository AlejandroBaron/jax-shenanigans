from typing import Any, Union

from jax.numpy import ndarray as jax_ndarray
from numpy import ndarray

array = Union[ndarray, jax_ndarray]


def is_jax_array(x: Any):
    return isinstance(x, jax_ndarray)
