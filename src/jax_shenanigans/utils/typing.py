from typing import Union

from jax.numpy import ndarray as jax_ndarray
from numpy import ndarray

array = Union[ndarray, jax_ndarray]
