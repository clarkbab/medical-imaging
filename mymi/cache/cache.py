import hashlib
import inspect
import json
import numpy as np
import os
import pandas as pd
import pickle
from pickle import UnpicklingError
import shutil
import time
from typing import Any, Callable, Dict, List, OrderedDict, Sequence, Tuple

from mymi import config
from mymi import logging

class Cache:
    Types = [
        Dict,
        np.ndarray,
        OrderedDict,
        pd.DataFrame
    ]
        
    def __init__(self):
        self._logging = False
        self._read_enabled = True
        self._write_enabled = True

    @property
    def logging(self) -> bool:
        return self._logging

    @logging.setter
    def logging(
        self,
        enabled: bool) -> None:
        self._logging = enabled

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
            seq_types = (list, np.ndarray)
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
            logging.error(e)
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
            logging.error(e)
            return

        # Check if cache key exists.
        if not self._key_exists(key):
            return

        # Start cache read.
        all_params = { 'type': data_type, **params }
        if self._logging:
            logging.info(f"Reading from cache with params '{all_params}'.")
        start = time.time()

        # Read data.
        try:
            data = None
            if data_type in (Dict, OrderedDict):
                data = self._read_dict(key)
            elif data_type == np.ndarray:
                data = self._read_numpy_array(key)
            elif data_type == pd.DataFrame:
                data = self._read_pandas_data_frame(key)
        except EOFError as e:
            logging.error(f"Caught 'EOFError' when reading cache key '{key}'.")
            logging.error(f"Error: {e}")
        except UnpicklingError as e:
            logging.error(f"Caught 'UnpicklingError' when reading cache key '{key}'.")
            logging.error(f"Error: {e}")
            return None

        # Log cache finish time and data size.
        if self._logging:
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
        # Check if cache read is enabled.
        if not self._write_enabled:
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
            logging.error(e)
            return

        # Start cache write.
        all_params = { 'type': data_type, **params }
        if self._logging:
            logging.info(f"Writing to cache with params '{all_params}'.")
        start = time.time()

        # Write data.
        size = None
        if data_type in (Dict, OrderedDict):
            size = self._write_dict(key, obj)
        elif data_type == np.ndarray:
            size = self._write_numpy_array(key, obj)
        elif data_type == pd.DataFrame:
            size = self._write_pandas_data_frame(key, obj)

        # Log cache finish time and data size.
        size_mb = size / (2 ** 20)
        if self._logging:
            logging.info(f"Complete [{size_mb:.3f}MB - {time.time() - start:.3f}s].")

    def _read_numpy_array(
        self,
        key: str) -> np.ndarray:
        """
        returns: the cached np.ndarray.
        args:
            key: the cache key.
        """
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'rb')
        return np.load(f)

    def _read_pandas_data_frame(
        self,
        key: str) -> pd.DataFrame:
        """
        returns: the cached pd.DataFrame.
        args:
            key: the cache key.
        """
        filepath = os.path.join(config.directories.cache, key)
        return pd.read_parquet(filepath)

    def _read_dict(
        self,
        key: str) -> dict:
        """
        returns: the cached dict.
        args:
            key: the cache key.
        """
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'rb')
        return pickle.load(f)

    def _read_string_numpy_array_pairs(
        self,
        key: str) -> List[Tuple[str, np.ndarray]]:
        """
        returns: the cached list of (str, np.ndarray) pairs.
        args:
            key: the cache key.
        """
        folder_path = os.path.join(config.directories.cache, key)
        name_array_pairs = []
        for name in os.listdir(folder_path):
            friendly_name = self._path_friendly(name)
            filepath = os.path.join(folder_path, friendly_name)
            f = open(filepath, 'rb')
            data = np.load(f)
            name_array_pairs.append((name, data))
        return name_array_pairs

    def _write_numpy_array(
        self,
        key: str,
        array: np.ndarray) -> int:
        """
        effect: writes the np.ndarray to the cache.
        returns: the size of the written cache in bytes.
        args:
            key: the cache key.
            array: the np.ndarray to cache.
        """
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'wb')
        np.save(f, array)
        return os.path.getsize(filepath) 

    def _write_pandas_data_frame(
        self,
        key: str,
        df: pd.DataFrame) -> int:
        """
        effect: writes the pd.DataFrame to the cache.
        returns: the size of the written cache in bytes.
        args:
            key: the cache key.
            df: the pd.DataFrame to cache.
        """
        filepath = os.path.join(config.directories.cache, key)
        df.to_parquet(filepath)
        return os.path.getsize(filepath) 

    def _write_dict(
        self,
        key: str,
        dictionary: dict) -> int:
        """
        effect: writes the dict to the cache.
        returns: the size of the written cache in bytes.
        args:
            key: the cache key.
            dictionary: the dict to cache.
        """
        filepath = os.path.join(config.directories.cache, key)
        f = open(filepath, 'wb')
        pickle.dump(dictionary, f)
        return os.path.getsize(filepath) 

    def _write_string_numpy_array_pairs(
        self,
        key: str,
        pairs: List[Tuple[str, np.ndarray]]) -> int:
        """
        effect: writes the list of (str, np.ndarray) pairs to the cache.
        returns: the size of the written cache in bytes.
        args:
            key: the cache key.
            pairs: the list of (str, np.ndarray) pairs.
        """
        folder_path = os.path.join(config.directories.cache, key)
        os.makedirs(folder_path, exist_ok=True)

        size = 0
        for name, data in pairs:
            friendly_name = self._path_friendly(name)
            filepath = os.path.join(folder_path, friendly_name)
            f = open(filepath, 'wb')
            np.save(f, data)
            size += os.path.getsize(filepath)
        return size

    def _path_friendly(
        self,
        string: str) -> str:
        """
        returns: a path-friendly string.
        args:
            string: the string.
        """
        # Replace special chars with underscores.
        chars = [' ', '.', '/', '\\', '[', ']']
        for char in chars:
            string = string.replace(char, '_')
        return string

    def function(
        self,
        fn: Callable) -> Callable:
        """
        returns: a wrapped function with result caching.
        args:
            fn: the function to cache.
        """
        # Get default kwargs.
        default_kwargs = {}
        argspec = inspect.getfullargspec(fn)
        if argspec.defaults:
            num_defaults = len(argspec.defaults)
            kwarg_names = argspec.args[-num_defaults:]
            kwarg_values = argspec.defaults
            for k, v in zip(kwarg_names, kwarg_values):
                default_kwargs[k] = v

        # Determine return type.
        sig = inspect.signature(fn)
        return_type = sig.return_annotation

        def wrapper(*args, **kwargs):
            # Merge kwargs with default kwargs for consistent cache keys when
            # arguments aren't passed.
            kwargs = { **default_kwargs, **kwargs }

            # Get 'clear_cache' param.
            clear_cache = kwargs.pop('clear_cache', False)

            # Create cache params.
            params = {
                'type': return_type,
                'method': fn.__name__
            }

            # Add args/kwargs.
            if len(args) > 0:
                params = { **params, 'args': args }
            params = { **params, **kwargs }

            # Clear cache.
            if clear_cache:
                self.delete(params)

            # Read from cache.
            result = self.read(params)
            if result is not None:
                return result

            # Add 'clear_cache' param back in if necessary to pass down.
            arg_names = inspect.getfullargspec(fn).args
            if 'clear_cache' in arg_names:
                kwargs['clear_cache'] = clear_cache

            # Call inner function.
            result = fn(*args, **kwargs)

            # Write data to cache.
            self.write(params, result)

            return result

        return wrapper

    def method(
        self,
        *attrs: Sequence[str]) -> Callable[[Callable], Callable]:
        """
        returns: a decorator providing result caching for instance methods.
        args:
            attrs: the instance attributes to include in cache parameters.
        """
        # Create decorator.
        def decorator(fn):
            # Get default kwargs.
            default_kwargs = {}
            argspec = inspect.getfullargspec(fn)
            if argspec.defaults:
                num_defaults = len(argspec.defaults)
                kwarg_names = argspec.args[-num_defaults:]
                kwarg_values = argspec.defaults
                for k, v in zip(kwarg_names, kwarg_values):
                    default_kwargs[k] = v

            # Determine return type.
            sig = inspect.signature(fn)
            return_type = sig.return_annotation

            def wrapper(inner_self, *args, **kwargs):
                # Merge kwargs with default kwargs for consistent cache keys when
                # arguments aren't passed.
                kwargs = {**default_kwargs, **kwargs}

                # Get 'clear_cache' param.
                clear_cache = kwargs.pop('clear_cache', False)

                # Create cache params.
                params = {
                    'type': return_type,
                    'method': fn.__name__
                }

                # Add specified instance attributes.
                for a in attrs:
                    # Handle nested attributes.
                    a_split = a.split('.')
                    if len(a_split) == 1:
                        params[a] = getattr(inner_self, a)
                    elif len(a_split) == 2:
                        params[a] = getattr(getattr(inner_self, a_split[0]), a_split[1])

                # Add args/kwargs.
                if len(args) > 0:
                    params = { **params, 'args': args }
                params = { **params, **kwargs }

                # Clear cache.
                if clear_cache:
                    self.delete(params)

                # Read from cache.
                result = self.read(params)
                if result is not None:
                    return result

                # Add 'clear_cache' param back in if necessary to pass down.
                arg_names = inspect.getfullargspec(fn).args

                if 'clear_cache' in arg_names:
                    kwargs['clear_cache'] = clear_cache

                # Call inner function.
                result = fn(inner_self, *args, **kwargs)

                # Write data to cache.
                self.write(params, result)

                return result

            return wrapper

        return decorator