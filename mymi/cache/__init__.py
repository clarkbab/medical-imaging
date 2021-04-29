from .cache import Cache
import logging
import os

# Create cache
cache = Cache()

def config(**kwargs):
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

def delete(*args):
    return cache.delete(*args)

def exists(*args):
    return cache.exists(*args)

def read(*args):
    return cache.read(*args)

def write(*args):
    return cache.write(*args)
