from collections import OrderedDict
from datetime import datetime
import glob
import gzip
import hashlib
import json
import logging
import numpy as np
from numpy import ndarray
import os
import pandas as pd
from pandas import DataFrame
import pickle
import pydicom as dicom
import shutil
import time
from typing import *

from mymi import config

class Cache:
    Types = [
        DataFrame,
        dict,
        ndarray,
        OrderedDict,
        Sequence[Tuple[str, ndarray]]
    ]
        
    def __init__(self):
        self._read_enabled = True
        self._write_enabled = True

    @property
    def read_enabled(self) -> bool:
        return self._read_enabled

    @read_enabled.setter
    def read_enabled(
        self,
        enabled: bool) -> None:
        self._read_enabled = enabled

    @property
    def write_enabled(self) -> bool:
        return self._write_enabled

    @write_enabled.setter
    def write_enabled(
        self,
        enabled: bool) -> None:
        self._write_enabled = enabled

    def _require_cache(fn: Callable) -> Callable:
        """
        returns: a wrapped function, ensuring cache exists.
        args:
            fn: the function to wrap.
        """
        def wrapper(self, *args, **kwargs):
            if not os.path.exists(config.directories.cache):
                os.makedirs(config.directories.cache)
            return fn(self, *args, **kwargs)
        return wrapper

    def _cache_key(
        self,
        params: dict) -> str:
        """
        returns: the hashed cache key.
        kwargs:
            params: the dictionary of cache parameters.
        """
        # Sort sequences for consistent cache keys.
        params = self._sort_sequences(params)

        # Convert any non-JSON-serialisable parameters.
        params = self._make_serialisable(params)

        # Create hash.
        hash = hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest() 

        return hash

    def _sort_sequences(
        self,
        params: dict) -> dict:
        """
        returns: a dict with sequences sorted.
        args:
            params: a dict.
        """
        # Create sorted params.
        sorted_params = {}
        for k, v in params.items():
            if type(v) in (tuple, list):
                v = list(sorted(v))
            sorted_params[k] = v
        
        return sorted_params

    def is_serialisable(
        self, 
        obj: Any) -> bool:
        """
        returns: whether the object is JSON-serialisable.
        args:
            obj: the object to inspect.
        """
        try:
            json.dumps(obj)
            return True
        except TypeError:
            return False

    def _make_serialisable(
        self,
        obj: Any) -> Any:
        """
        returns: a dict that is JSON-serialisable.
        args:
            obj: the object to serialise.
        """
        if self.is_serialisable(obj):
            return obj
        else:
            # Handle 'sequence' types.
            seq_types = (list, ndarray)
            if type(obj) in seq_types:
                obj = [self._make_serialisable(o) for o in obj]
                return obj

            # Handle 'dict' type.
            if type(obj) == dict:
                for k, v in obj.items():
                    obj[k] = self._make_serialisable(v) 
                return obj

            # Handle custom types.
            if hasattr(obj, 'cache_key'):
                return obj.cache_key()

        raise ValueError(f"Cache key can't contain type '{type(obj)}', must be JSON-serialisable or implement 'cache_key' method.")

    @_require_cache
    def delete(
        self,
        params: dict) -> None:
        """
        effect: deletes the cached object.
        args:
            params: the params of the cached object.
        """
        # Remove 'type' for consistency.
        params = params.copy()
        _ = params.pop('type', None)

        # Get cache key string.
        try:
            key = self._cache_key(params)
        except ValueError as e:
            # Types can signal that they're uncacheable by raising a 'ValueError', e.g. 'RandomResample'.
            logging.info(e)
            return

        # Check if cache key exists.
        if not self._key_exists(key):
            return

        # Delete the file/folder.
        key_path = os.path.join(config.directories.cache, key)
        if os.path.isdir(key_path):
            shutil.rmtree(key_path)
        else:
            os.remove(key_path)

    def _key_exists(
        self, 
        key: str) -> bool:
        """
        returns: whether the key exists.
        args:
            key: the key to search for.
        """
        # Search for file by key.
        key_path = os.path.join(config.directories.cache, key)
        if os.path.exists(key_path):
            return True
        else:
            return False

    @_require_cache
    def read(
        self,
        params: dict) -> Any:
        """
        returns: the cached object.
        args:
            params: the dict of cache params.
        """
        logging.info(f"Reading from cache with params '{params}'.")

        # Check if cache read is enabled.
        if not self._read_enabled:
            return

        # Get the data type.
        params = params.copy()
        data_type = params.pop('type', None)
        if data_type is None:
            raise ValueError(f"Cache params must include 'type', got '{params}'.")
        if data_type not in self.Types:
            raise ValueError(f"Cache type '{data_type}' not recognised, allowed types '{self.Types}'.")
        
        # Get cache key string.
        try:
            key = self._cache_key(params)
        except ValueError as e:
            # Types can signal that they're uncacheable by raising a 'ValueError', e.g. 'RandomResample'.
            logging.info(e)
            return

        # Check if cache key exists.
        if not self._key_exists(key):
            return

        # Log cache read start.
        start = time.time()

        # Read data.
        data = None
        if data_type == DataFrame:
            data = self._read_pandas_data_frame(key)
        elif data_type == dict or data_type == OrderedDict:
            data = self._read_dict(key)
        elif data_type == ndarray:
            data = self._read_numpy_array(key)
        elif data_type == Sequence[Tuple[str, ndarray]]:
            data = self._read_string_numpy_array_pairs(key)

        # Log cache finish time and data size.
        logging.info(f"Complete [{time.time() - start:.3f}s].")

        return data

    @_require_cache
    def write(
        self,
        params: dict,
        obj: Any) -> None:
        """
        effect: writes object to cache.
        args:
            params: cache parameters for the object.
            obj: the object to cache.
        """
        logging.info(f"Writing to cache with params '{params}'.")

        # Check if cache read is enabled.
        if not self._write_enabled:
            return

        # Get the data type.
        params = params.copy()
        data_type = params.pop('type', None)
        if data_type is None:
            raise ValueError(f"Cache params must include 'type', got '{params}'.")
        if data_type not in self.Types:
            raise ValueError(f"Cache type '{data_type}' not recognised, allowed types '{Types}'.")
        
        # Get cache key string.
        try:
            key = self._cache_key(params)
        except ValueError as e:
            # Types can signal that they're uncacheable by raising a 'ValueError', e.g. 'RandomResample'.
            logging.info(e)
            return

        # Log cache write start.
        start = time.time()

        # Write data.
        size = None
        if data_type == DataFrame:
            size = self._write_pandas_data_frame(key, obj)
        elif data_type == dict or data_type == OrderedDict:
            size = self._write_dict(key, obj)
        elif data_type == ndarray:
            size = self._write_numpy_array(key, obj)
        elif data_type == Sequence[Tuple[str, ndarray]]:
            size = self._write_string_numpy_array_pairs(key, obj)

        # Log cache finish time and data size.
        size_mb = size / (2 ** 20)
        logging.info(f"Complete [{size_mb:.3f}MB - {time.time() - start:.3f}s].")

    def _read_numpy_array(self, key):
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'rb')
        return np.load(f)

    def _read_pandas_data_frame(self, key):
        filepath = os.path.join(config.directories.cache, key)
        return pd.read_parquet(filepath)

    def _read_dict(self, key):
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'rb')
        return pickle.load(f)

    def _read_string_numpy_array_pairs(self, key):
        folder_path = os.path.join(config.directories.cache, key)
        name_array_pairs = []
        for name in os.listdir(folder_path):
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'rb')
            data = np.load(f)
            name_array_pairs.append((name, data))
        
        return name_array_pairs

    def _read_file_dataset_list(self, key):
        folder_path = os.path.join(config.directories.cache, key)
        fds = []
        for name in sorted(os.listdir(folder_path)):
            filepath = os.path.join(folder_path, name)
            fd = dicom.read_file(filepath)
            fds.append(fd)
        
        return fds

    def _write_numpy_array(self, key, array):
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'wb')
        np.save(f, array)
        return os.path.getsize(filepath) 

    def _write_pandas_data_frame(self, key, df):
        filepath = os.path.join(config.directories.cache, key)
        df.to_parquet(filepath)
        return os.path.getsize(filepath) 

    def _write_dict(self, key, dictionary):
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'wb')
        pickle.dump(dictionary, f)
        return os.path.getsize(filepath) 

    def _write_string_numpy_array_pairs(self, key, pairs):
        folder_path = os.path.join(config.directories.cache, key)
        os.makedirs(folder_path, exist_ok=True)

        size = 0
        for name, data in pairs:
            filepath = os.path.join(folder_path, name)
            f = open(filepath, 'wb')
            np.save(f, data)
            size += os.path.getsize(filepath)

        return size

    def _write_file_dataset_list(self, key, fds):
        folder_path = os.path.join(config.directories.cache, key)
        os.makedirs(folder_path, exist_ok=True)

        size = 0
        for i, fd in enumerate(fds):
            filepath = os.path.join(folder_path, f"{i}.dcm")
            fd.save_as(filepath)
            size += os.path.getsize(filepath)

        return size
            