import inspect
import logging
import os
from typing import *

from .cache import Cache

# Create cache
cache = Cache()

def delete(*args, **kwargs):
    return cache.delete(*args, **kwargs)

def exists(*args, **kwargs):
    return cache.exists(*args, **kwargs)

def read(*args, **kwargs):
    return cache.read(*args, **kwargs)

def write(*args, **kwargs):
    return cache.write(*args, **kwargs)

def cached_method(*attrs: Sequence[str]) -> Callable[[Callable], Callable]:
    """
    effect: caches the result of the wrapped function.
    args:
        attrs: the instance attributes to include in cache parameters.
    """
    # Create decorator.
    def decorator(fn):
        # Determine return type.
        sig = inspect.signature(fn)
        return_type = sig.return_annotation

        def wrapper(self, *args, **kwargs):
            # Get 'clear_cache' param.
            clear_cache = kwargs.pop('clear_cache', False)

            # Create cache params.
            params = {
                'type': return_type,
                'method': fn.__name__
            }

            # Add specified instance attributes.
            for a in attrs:
                params[a] = getattr(self, a)

            # Add args/kwargs.
            params = {**params, **kwargs}

            # Clear cache.
            if clear_cache:
                cache.delete(params)

            # Read from cache.
            result = cache.read(params)
            if result is not None:
                return result

            # Add 'clear_cache' param back in if necessary to pass down.
            arg_names = inspect.getfullargspec(fn).args
            if 'clear_cache' in arg_names:
                kwargs['clear_cache'] = clear_cache

            # Call inner function.
            result = fn(self, *args, **kwargs)

            # Write data to cache.
            cache.write(params, result)

            return result

        return wrapper

    return decorator

def config(**kwargs: dict) -> None:
    """
    effect: configures the cache.
    kwargs:
        path: the path to the cache.
        read_enabled: enables cache read.
        write_enabled: enables cache write.
    """
    # Set cache path.
    path = kwargs.pop('path', None)
    if path is not None:
        cache.path = path

    # Set enabled flags.
    cache.read_enabled = kwargs.pop('read_enabled', True)
    cache.write_enabled = kwargs.pop('write_enabled', True)
