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
    # Set configured flag.
    cache.set_configured()

    # Configure enabled.
    enabled = kwargs.pop('enabled', None)
    if not enabled:
        cache.read_enabled = False
        cache.write_enabled = False
    else:
        cache.read_enabled = kwargs.pop('read_enabled', True)
        cache.write_enabled = kwargs.pop('write_enabled', True)

def exists(key):
    if not cache.configured:
        config()

    return cache.exists(key)

def read(key, type):
    if not cache.configured:
        config()

    return cache.read(key, type)

def write(key, obj, type):
    if not cache.configured:
        config()

    return cache.write(key, obj, type)

def read_enabled():
    if not cache.configured:
        config()

    return cache.read_enabled

def write_enabled():
    if not cache.configured:
        config()

    return cache.write_enabled
