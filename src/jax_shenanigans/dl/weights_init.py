from typing import Tuple

from jax import random


def random_init(
    shape: Tuple[int], key: random.KeyArray, distribution: str = "uniform", **kwargs
):
    r_sample = getattr(random, distribution)
    return r_sample(key=key, shape=shape, **kwargs)
