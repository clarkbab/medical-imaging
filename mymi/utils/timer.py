from contextlib import contextmanager
import pandas as pd
from time import time
from typing import *

from .io import save_csv
from .pandas import append_row

# Time for each 'recorded' event is stored in a row of the CSV.
# Additional columns can be populated using 'data'.
class Timer:
    def __init__(
        self,
        columns: Dict[str, str] = {}):
        self.__cols = columns
        self.__cols['time'] = float
        self.__df = pd.DataFrame(columns=self.__cols.keys())

    @contextmanager
    def record(
        self,
        data: Dict[str, Union[str, int, float]] = {},
        enabled: bool = True):
        try:
            if enabled:
                start = time()
            yield None
        finally:
            if enabled:
                data['time'] = time() - start
                self.__df = append_row(self.__df, data)

    def save(self, filepath):
        df = self.__df.astype(self.__cols)
        save_csv(df, filepath, overwrite=True)
 