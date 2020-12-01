from datetime import datetime
import glob
import gzip
import hashlib
import json
import numpy as np
import os
import pandas as pd

class DataCache:
    def __init__(self, root):
        """
        root: the root folder for the cache.
        """
        self.root = root

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
        if type == 'array':
            return self.read_array(key)
        elif type == 'dataframe':
            return self.read_dataframe(key)
        elif type == 'name-array-pairs':
            return self.read_name_array_pairs(key)
        else:
            # TODO: raise error.
            print('unrecognised cache type')
            return None

    def write(self, key, obj, type):
        """
        key: a dict of terms that define a unique cache item.
        obj: the object to cache.
        type: the object type.
        """
        if type == 'array':
            self.write_array(key, obj)
        elif type == 'dataframe':
            self.write_dataframe(key, obj)
        elif type == 'name-array-pairs':
            self.write_name_array_pairs(key, obj)
        else:
            # TODO: raise error.
            print('unrecognised cache type')

    def read_array(self, key):
        """
        key: the cache key string.
        """
        # Read data.
        filepath = os.path.join(self.root, self.cache_key(key))
        f = gzip.GzipFile(filepath, 'r')
        data = np.load(f)

        return data

    def read_dataframe(self, key):
        """
        key: the cache key string.
        """
        # Read data.
        filepath = os.path.join(self.root, self.cache_key(key))
        data = pd.read_parquet(filepath)

        return data

    def read_name_array_pairs(self, key):
        """
        returns: a list of (name, array) pairs.
        key: the cache key string.
        """
        # Read data.
        folder_path = os.path.join(self.root, self.cache_key(key))

        name_array_pairs = []
        for name in os.listdir(folder_path):
            filepath = os.path.join(folder_path, name)
            f = gzip.GzipFile(filepath, 'r')
            data = np.load(f)
            name_array_pairs.append((name, data))

        return name_array_pairs

    def write_array(self, key, array):
        """
        key: the cache key string.
        df: the dataframe.
        """
        # Write data.
        filepath = os.path.join(self.root, self.cache_key(key))
        f = gzip.GzipFile(filepath, 'w')
        np.save(f, array)
        f.close()

    def write_dataframe(self, key, df):
        """
        key: the cache key string.
        df: the dataframe.
        """
        # Write data.
        filename = os.path.join(self.root, self.cache_key(key))
        df.to_parquet(filename)

    def write_name_array_pairs(self, key, pairs):
        """
        key: the cache key string.
        pairs: the list of (name, array) pairs.
        """
        # Write data.
        folder_path = os.path.join(self.root, self.cache_key(key))
        os.makedirs(folder_path, exist_ok=True)

        for name, data in pairs:
            filename = os.path.join(folder_path, name)
            f = gzip.GzipFile(filename, 'w')
            np.save(f, data)

    def cache_key(self, key):
        """
        key: a dict of cache parameters.
        """
        return hashlib.sha1(json.dumps(key).encode('utf-8')).hexdigest() 


            