from .cache import Cache
import logging
import os

# Create cache
cache = Cache()

def config(**kwargs):
    """
    effect: configures the cache.
    kwargs:
        path: path to the cache.
        read_enabled: enables cache read.
        write_enabled: enables cache write.
    """
    # Configure enabled.
    enabled = kwargs.pop('enabled', None)
    if enabled is not None:
        if not enabled:
            cache.read_enabled = False
            cache.write_enabled = False
        else:
            cache.read_enabled = kwargs.pop('read_enabled', True)
            cache.write_enabled = kwargs.pop('write_enabled', True)
    else:
        if 'MYMI_CACHE_ENABLED' in os.environ:
            enabled = os.environ['MYMI_CACHE_ENABLED']
            if enabled == 'false':
                cache.read_enabled = False
                cache.write_enabled = False
            elif enabled == 'true':
                cache.read_enabled = kwargs.pop('read_enabled', True)
                cache.write_enabled = kwargs.pop('write_enabled', True)
            else:
                raise ValueError(f"Invalid value '{enabled}' for 'MYMI_CACHE_ENABLED', expected 'true' or 'false'.")

    # Configure path.
    path = kwargs.pop('path', None)
    if path is not None:
        if os.path.exists(path):
            cache.path = path
        else:
            raise ValueError(f"Cache path '{path}', set via config doesn't exist.")
    else:
        if 'MYMI_CACHE' in os.environ:
            env_path = os.environ['MYMI_CACHE']
            if os.path.exists(env_path):
                cache.path = env_path
            else:
                raise ValueError(f"Cache path '{env_path}', set via env var doesn't exist.")
        else:
            default_path = os.path.join(os.sep, 'tmp', 'mymi', 'cache')
            if os.path.exists(default_path):
                os.makedirs(default_path, exist_ok=True)
                cache.path = default_path
                if path is not None or 'MYMI_CACHE' in os.environ:
                    logging.warning(f"Cache using default path '{default_path}'.")
            else:
                raise ValueError(f"Default cache path '{default_path}' doesn't exist.")

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
