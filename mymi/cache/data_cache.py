from datetime import datetime
import glob
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
        else:
            # TODO: raise error.
            print('unrecognised cache type')

    def read_array(self, key):
        """
        key: the cache key string.
        """
        # Read data.
        data_filename = 'data.npy'
        cache_path = os.path.join(self.root, self.cache_key(key))
        data_path = os.path.join(cache_path, data_filename)
        data = np.load(data_path)

        # Write last read date.
        read_filename = 'read.txt'
        read_path = os.path.join(cache_path, read_filename)
        read_file = open(read_path, 'w')
        read_file.write(str(datetime.now()))
        read_file.close()

        return data

    def read_dataframe(self, key):
        """
        key: the cache key string.
        """
        # Read data.
        data_filename = 'data.csv'
        cache_path = os.path.join(self.root, self.cache_key(key))
        data_path = os.path.join(cache_path, data_filename)
        data = pd.read_csv(data_path, index_col=0)

        # Write last read date.
        read_filename = 'read.txt'
        read_path = os.path.join(cache_path, read_filename)
        read_file = open(read_path, 'w')
        read_file.write(str(datetime.now()))
        read_file.close()

        return data

    def write_array(self, key, array):
        """
        key: the cache key string.
        df: the dataframe.
        """
        # Write data.
        data_filename = 'data.npy'
        cache_path = os.path.join(self.root, self.cache_key(key))
        data_path = os.path.join(cache_path, data_filename)
        os.makedirs(cache_path, exist_ok=True)
        np.save(data_path, array)

        # Write out write date.
        write_filename = 'write.txt'
        write_path = os.path.join(cache_path, write_filename)
        write_file = open(write_path, 'w')
        write_file.write(str(datetime.now()))
        write_file.close()

    def write_dataframe(self, key, df):
        """
        key: the cache key string.
        df: the dataframe.
        """
        # Write data.
        data_filename = 'data.csv'
        cache_path = os.path.join(self.root, self.cache_key(key))
        data_path = os.path.join(cache_path, data_filename)
        os.makedirs(cache_path, exist_ok=True)
        df.to_csv(data_path)

        # Write out write date.
        write_filename = 'write.txt'
        write_path = os.path.join(cache_path, write_filename)
        write_file = open(write_path, 'w')
        write_file.write(str(datetime.now()))
        write_file.close()

    def cache_key(self, key):
        """
        key: a dict of cache parameters.
        """
        return hashlib.sha1(json.dumps(key).encode('utf-8')).hexdigest() 


            