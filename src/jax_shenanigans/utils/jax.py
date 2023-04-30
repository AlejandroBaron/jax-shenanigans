import jax.numpy as jnp


def fill_diagonal(x: jnp.ndarray, value: int):
    return x.at[jnp.diag_indices(len(x))].set(value)


def is_between(x: jnp.ndarray, a: int, b: int):
    return (a <= b) & (x <= b)
