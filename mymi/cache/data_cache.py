from datetime import datetime
import glob
import gzip
import hashlib
import json
import numpy as np
import os
import pandas as pd
import time

class DataCache:
    def __init__(self, root, verbose=False):
        """
        root: the root folder for the cache.
        """
        self.root = root
        self.verbose = verbose

    def exists(self, key):
        """
        key: cache key to look for.
        """
        # Search for file by key.
        cache_path = os.path.join(self.root, self.cache_key(key))
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
        if self.verbose:
            print(f"Reading {type} from cache with key '{key}'.")

        # Read data.
        data, size = None, None
        if type == 'array':
            data, size = self.read_array(key)
        elif type == 'dataframe':
            data, size = self.read_dataframe(key)
        elif type == 'name-array-pairs':
            data, size = self.read_name_array_pairs(key)
        else:
            # TODO: raise error.
            print('unrecognised cache type')

        # Log cache finish time.
        if self.verbose:
            # Convert size to megabytes.
            size_mb = size / (2 ** 20)
            print(f"Complete [{size_mb:.3f}MB - {time.time() - start:.3f}s].")

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
        if self.verbose:
            print(f"Writing {type} to cache with key '{key}'.")

        # Write data.
        size = None
        if type == 'array':
            size = self.write_array(key, obj)
        elif type == 'dataframe':
            size = self.write_dataframe(key, obj)
        elif type == 'name-array-pairs':
            size = self.write_name_array_pairs(key, obj)
        else:
            # TODO: raise error.
            print('unrecognised cache type')

        # Log cache finish time.
        if self.verbose:
            # Convert size to megabytes.
            size_mb = size / (2 ** 20)
            print(f"Complete [{size_mb:.3f}MB - {time.time() - start:.3f}s].")

    def read_array(self, key):
        """
        key: the cache key string.
        """
        # Read data.
        filepath = os.path.join(self.root, self.cache_key(key))
        f = open(filepath, 'rb')
        data = np.load(f)

        # Get file size.
        size = os.path.getsize(filepath)

        return data, size

    def read_dataframe(self, key):
        """
        key: the cache key string.
        """
        # Read data.
        filepath = os.path.join(self.root, self.cache_key(key))
        data = pd.read_parquet(filepath)

        # Get file size.
        size = os.path.getsize(filepath)

        return data, size

    def read_name_array_pairs(self, key):
        """
        returns: a list of (name, array) pairs.
        key: the cache key string.
        """
        # Read data.
        folder_path = os.path.join(self.root, self.cache_key(key))

        # Load data.
        name_array_pairs = []
        size = 0
        for name in os.listdir(folder_path):
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'rb')
            data = np.load(f)
            name_array_pairs.append((name, data))
            size += os.path.getsize(filepath)

        return name_array_pairs, size

    def write_array(self, key, array):
        """
        returns: the size of the cached data.
        key: the cache key string.
        df: the dataframe.
        """
        # Write data.
        filepath = os.path.join(self.root, self.cache_key(key))
        f = open(filepath, 'wb')
        np.save(f, array)

        return os.path.getsize(filepath) 

    def write_dataframe(self, key, df):
        """
        key: the cache key string.
        df: the dataframe.
        """
        # Write data.
        filepath = os.path.join(self.root, self.cache_key(key))
        df.to_parquet(filepath)

        return os.path.getsize(filepath) 

    def write_name_array_pairs(self, key, pairs):
        """
        key: the cache key string.
        pairs: the list of (name, array) pairs.
        """
        # Write data.
        folder_path = os.path.join(self.root, self.cache_key(key))
        os.makedirs(folder_path, exist_ok=True)

        # Save cache files.
        size = 0
        for name, data in pairs:
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'wb')
            np.save(f, data)
            size += os.path.getsize(filepath)

        return size

    def cache_key(self, key):
        """
        key: a dict of cache parameters.
        """
        return hashlib.sha1(json.dumps(key).encode('utf-8')).hexdigest() 


            