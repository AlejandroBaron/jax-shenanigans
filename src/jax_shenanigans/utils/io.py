from typing import Union

import jax.numpy as jnp
import numpy as np


def load_txt(
    path: str, as_jax: bool = True, **kwargs
) -> Union[np.ndarray, jnp.ndarray]:
    """Loads a txt rowwise."""
    txt = np.loadtxt(path, **kwargs)
    if as_jax:
        txt = jnp.asarray(txt)
    return txt
