from functools import wraps
from time import time as time

from loguru import logger


def with_timing(return_t: bool = False, log: bool = True):
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
