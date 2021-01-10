from .cache import Cache
import logging
import os

# Create cache
cache = Cache()

def config(**kwargs):
    """
    effect: configures the cache.
    path: path to the cache.
    read_enabled: enables cache read.
    write_enabled: enables cache write.
    """
    # Configure enabled.
    cache.read_enabled = kwargs.pop('read_enabled', True)
    cache.write_enabled = kwargs.pop('write_enabled', True)

    # Configure path.
    path = kwargs.pop('path', None)
    if path is not None:
        if os.path.exists(path):
            cache.path = path
        else:
            logging.warning(f"Cache path '{path}', set via config doesn't exist.")
    if cache.path is None and 'MYMI_CACHE' in os.environ:
        env_path = os.environ['MYMI_CACHE']
        if os.path.exists(env_path):
            cache.path = env_path
        else:
            logging.warning(f"Cache path '{env_path}', set via env var doesn't exist.")
    if cache.path is None:
        default_path = os.path.join(os.sep, 'tmp', 'mymi', 'cache')
        os.makedirs(default_path, exist_ok=True)
        cache.path = default_path
        if path is not None or 'MYMI_CACHE' in os.environ:
            logging.warning(f"Cache using default path '{default_path}'.")

def exists(key):
    if cache.path is None:
        config()

    return cache.exists(key)

def read(key, type):
    if cache.path is None:
        config()

    return cache.read(key, type)

def write(key, obj, type):
    if cache.path is None:
        config()

    return cache.write(key, obj, type)

def read_enabled():
    if cache.path is None:
        config()

    return cache.read_enabled

def write_enabled():
    if cache.path is None:
        config()

    return cache.write_enabled
