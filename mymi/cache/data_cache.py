import glob
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
        key_path = os.path.join(self.root, *key.split(':'))
        glob_str = f"{key_path}.*"
        if glob.glob(glob_str):
            return True
        else:
            return False

    def read(self, key, type):
        """
        key: the cache key string.
        type: the object type.
        """
        if type == 'dataframe':
            return self.read_dataframe(key)
        else:
            # TODO: raise error.
            print('unrecognised cache type')
            return None

    def write(self, key, obj, type):
        """
        key: the cache key string.
        obj: the object to cache.
        type: the object type.
        """
        if type == 'dataframe':
            self.write_dataframe(key, obj)
        else:
            # TODO: raise error.
            print('unrecognised cache type')

    def read_dataframe(self, key):
        """
        key: the cache key string.
        """
        key_path = os.path.join(self.root, *key.split(':'))
        file_path = f"{key_path}.csv"
        return pd.read_csv(file_path, index_col=0)

    def write_dataframe(self, key, df):
        """
        key: the cache key string.
        df: the dataframe.
        """
        key_path = os.path.join(self.root, *key.split(':'))
        file_path = f"{key_path}.csv"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path)

            