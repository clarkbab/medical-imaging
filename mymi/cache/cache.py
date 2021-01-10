from datetime import datetime
import glob
import gzip
import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import time

class Cache:
    def __init__(self):
        self._path = None
        self._read_enabled = None
        self._write_enabled = None

    @property
    def read_enabled(self):
        return self._read_enabled

    @read_enabled.setter
    def read_enabled(self, enabled):
        self._read_enabled = enabled

    @property
    def enabled_read(self):
        return self._read_enabled

    @property
    def write_enabled(self):
        return self._write_enabled

    @write_enabled.setter
    def write_enabled(self, enabled):
        self._write_enabled = enabled

    @property
    def enabled_write(self):
        return self._write_enabled

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        """
        path: path to the cache, should be a string and should exist.
        """
        if not isinstance(path, str):
            raise ValueError(f"Cache path should be string, got '{type(path)}'.")
        if not os.path.exists(path):
            raise ValueError(f"Cache path doesn't exist '{path}'.")
        self._path = path

    def cache_key(self, key):
        """
        key: a dict of cache parameters.
        """
        if not isinstance(key, dict):
            raise ValueError(f"Cache key should be dict, got '{type(key)}'.")

        return hashlib.sha1(json.dumps(key).encode('utf-8')).hexdigest() 

    def exists(self, key):
        """
        key: cache key to look for.
        """
        # Search for file by key.
        cache_path = os.path.join(self._path, self.cache_key(key))
        if os.path.exists(cache_path):
            return True
        else:
            return False

    def read(self, key, type):
        """
        key: the cache key string.
        type: the object type.
        """
        # Log cache read start.
        start = time.time()
        logging.info(f"Reading {type} from cache with key '{key}'.")

        # Read data.
        data = None
        if type == 'array':
            data = self.read_array(key)
        elif type == 'dataframe':
            data = self.read_dataframe(key)
        elif type == 'name-array-pairs':
            data = self.read_name_array_pairs(key)
        else:
            raise ValueError(f"Unrecognised cache type '{type}'.")

        # Log cache finish time and data size.
        logging.info(f"Complete - {time.time() - start:.3f}s].")

        return data

    def write(self, key, obj, type):
        """
        effect: creates cached data on disk.
        key: a dict of terms that define a unique cache item.
        obj: the object to cache.
        type: the object type.
        """
        # Log cache write start.
        start = time.time()
        logging.info(f"Writing {type} to cache with key '{key}'.")

        # Write data.
        size = None
        if type == 'array':
            size = self.write_array(key, obj)
        elif type == 'dataframe':
            size = self.write_dataframe(key, obj)
        elif type == 'name-array-pairs':
            size = self.write_name_array_pairs(key, obj)
        else:
            raise ValueError(f"Unrecognised cache type '{type}'.")

        # Log cache finish time and data size.
        size_mb = size / (2 ** 20)
        logging.info(f"Complete [{size_mb:.3f}MB - {time.time() - start:.3f}s].")

    def read_array(self, key):
        filepath = os.path.join(self._path, self.cache_key(key))
        f = open(filepath, 'rb')
        return np.load(f)

    def read_dataframe(self, key):
        filepath = os.path.join(self._path, self.cache_key(key))
        return pd.read_parquet(filepath)

    def read_name_array_pairs(self, key):
        folder_path = os.path.join(self._path, self.cache_key(key))
        name_array_pairs = []
        for name in os.listdir(folder_path):
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'rb')
            data = np.load(f)
            name_array_pairs.append((name, data))
        
        return name_array_pairs

    def write_array(self, key, array):
        filepath = os.path.join(self._path, self.cache_key(key))
        f = open(filepath, 'wb')
        np.save(f, array)
        return os.path.getsize(filepath) 

    def write_dataframe(self, key, df):
        filepath = os.path.join(self._path, self.cache_key(key))
        df.to_parquet(filepath)
        return os.path.getsize(filepath) 

    def write_name_array_pairs(self, key, pairs):
        folder_path = os.path.join(self._path, self.cache_key(key))
        os.makedirs(folder_path, exist_ok=True)

        size = 0
        for name, data in pairs:
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'wb')
            np.save(f, data)
            size += os.path.getsize(filepath)

        return size
            