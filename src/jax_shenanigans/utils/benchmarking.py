from functools import wraps
from time import time as time

import jax.numpy as jnp
from jax import random
from loguru import logger


def with_timing(return_t: bool = False, log: bool = True):
    """Decorator that times a function.

    It allows the user to retrieve or log the timing

    Args:
        return_t (bool, optional): If true, returns a (time, result) tuple
        log (bool, optional): If true, logs the time through loguru's logger
    """

    def decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            t0 = time()
            result = f(*args, **kwargs)
            tdiff = time() - t0
            if log:
                logger.info(f"{f.__name__} took {tdiff:.5f}s")
            return (result, tdiff) if return_t else result

        return wrap

    return decorator


def random_linear_setup(n, p, key, o=1):
    X = random.uniform(key, (n, p - 1))
    X = jnp.concatenate([jnp.ones((n, 1)), X], axis=1)
    B = random.randint(key, (p, o), 0, 10)

    # Internally, random.normal uses a uniform generator.
    # If the same key is used, the correlation with a
    # normal distribution is almost 1
    _, e_key = random.split(key)
    epsilon = random.normal(e_key, (n, 1)) * 0.3
    y = X @ B + epsilon
    return B, X, y
