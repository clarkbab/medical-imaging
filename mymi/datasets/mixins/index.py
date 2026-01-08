import pandas as pd
from typing import *

class IndexMixin:
    def __init__(
        self,
        *args,
        **kwargs) -> None:
        if 'index' in kwargs:
            self._index = kwargs.pop('index')
        if 'index_policy' in kwargs:
            self._index_policy = kwargs.pop('index_policy')
        super().__init__(*args, **kwargs)

    def index(
        self,
        **filters) -> pd.DataFrame:
        def index_fn(_, **filters) -> pd.DataFrame:
            index = self._index.copy()
            for k, v in filters.items():
                index = index[index[k] == v]
            return index
        index_fn = self.__class__.ensure_loaded(index_fn) if hasattr(self.__class__, 'ensure_loaded') else index_fn
        return index_fn(self, **filters)

    @property
    def index_policy(self) -> Dict[str, Any]:
        return self.__class__.ensure_loaded(lambda _: self._index_policy) if hasattr(self.__class__, 'ensure_loaded') else self._index_policy

class IndexWithErrorsMixin(IndexMixin):
    def index_errors(
        self,
        **filters) -> pd.DataFrame:
        # 'index_errors_fn' is not bound to the instance, so it's 'self' won't have access to '_index'.
        def index_errors_fn(_, **filters) -> pd.DataFrame:
            index_errors = self._index_errors.copy()
            for k, v in filters.items():
                index_errors = index_errors[index_errors[k] == v]
            return index_errors
        index_errors_fn = self.__class__.ensure_loaded(index_errors_fn) if hasattr(self.__class__, 'ensure_loaded') else index_errors_fn
        return index_errors_fn(self, **filters)
        