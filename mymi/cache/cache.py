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

data_dir = os.environ['MYMI_DATA']
CACHE_ROOT = os.path.join(data_dir, 'cache') 

class Cache:
    def __init__(self):
        self._configured = False
        self._read_enabled = None
        self._write_enabled = None

    @property
    def configured(self):
        return self._configured

    def set_configured(self):
        self._configured = True

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

    def cache_key(self, params):
        """
        params: a dict of cache parameters.
        """
        if not isinstance(params, dict):
            raise ValueError(f"Cache params should be dict, got '{type(params)}'.")

        # Handle special types so that dict is serialisable.
        params = self.make_serialisable(params)

        return hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest() 

    def is_serialisable(self, obj):
        """
        returns: True if the object is JSON serialisable, False otherwise.
        obj: the object to convert.
        """
        try:
            json.dumps(obj)
            return True
        except TypeError:
            return False

    def make_serialisable(self, obj):
        """
        returns: an object that is JSON serialisable.
        obj: the object to convert.
        """
        # Check serialisability.
        if not self.is_serialisable(obj):
            # Handle dict.
            if isinstance(obj, dict):
                for k, v in obj.items():
                    obj[k] = self.make_serialisable(v) 
            # Handle known types.
            elif isinstance(obj, list) or isinstance(obj, np.ndarray):
                obj = [self.make_serialisable(o) for o in obj]
            # Handle custom types.
            elif hasattr(obj, 'cache_key'):
                obj = obj.cache_key()
            else:
                raise ValueError(f"Object {obj} can't be passed as cache key, must be serialisable or implement 'cache_key' method.")

        return obj

    def exists(self, key):
        """
        key: cache key to look for.
        """
        # Search for file by key.
        cache_path = os.path.join(CACHE_ROOT, key) 
        if os.path.exists(cache_path):
            return True
        else:
            return False

    def read(self, params, type):
        """
        params: the cache key dict.
        type: the object type.
        """
        # Check if cache read is enabled.
        if not self.read_enabled:
            return None
        
        # Get cache key string.
        try:
            key = self.cache_key(params)
        except ValueError as e:
            # Types can signal that they're uncacheable by raising a 'ValueError', e.g. 'RandomResample'.
            logging.info(e)
            return None

        # Check if cache key exists.
        if not self.exists(key):
            return None

        # Log cache read start.
        start = time.time()
        logging.info(f"Reading {type} from cache with params '{params}'.")

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
        logging.info(f"Complete [{time.time() - start:.3f}s].")

        return data

    def write(self, params, obj, type):
        """
        effect: creates cached data on disk.
        params: a dict of terms that define a unique cache item.
        obj: the object to cache.
        type: the object type.
        """
        # Check if cache read is enabled.
        if not self.write_enabled:
            return None
        
        # Get cache key string.
        try:
            key = self.cache_key(params)
        except ValueError as e:
            # Types can signal that they're uncacheable by raising a 'ValueError', e.g. 'RandomResample'.
            logging.info(e)
            return None

        # Log cache write start.
        start = time.time()
        logging.info(f"Writing {type} to cache with params '{params}'.")

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
        filepath = os.path.join(CACHE_ROOT, key)
        f = open(filepath, 'rb')
        return np.load(f)

    def read_dataframe(self, key):
        filepath = os.path.join(CACHE_ROOT, key)
        return pd.read_parquet(filepath)

    def read_name_array_pairs(self, key):
        folder_path = os.path.join(CACHE_ROOT, key)
        name_array_pairs = []
        for name in os.listdir(folder_path):
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'rb')
            data = np.load(f)
            name_array_pairs.append((name, data))
        
        return name_array_pairs

    def write_array(self, key, array):
        filepath = os.path.join(CACHE_ROOT, key)
        f = open(filepath, 'wb')
        np.save(f, array)
        return os.path.getsize(filepath) 

    def write_dataframe(self, key, df):
        filepath = os.path.join(CACHE_ROOT, key)
        df.to_parquet(filepath)
        return os.path.getsize(filepath) 

    def write_name_array_pairs(self, key, pairs):
        folder_path = os.path.join(CACHE_ROOT, key)
        os.makedirs(folder_path, exist_ok=True)

        size = 0
        for name, data in pairs:
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'wb')
            np.save(f, data)
            size += os.path.getsize(filepath)

        return size
            