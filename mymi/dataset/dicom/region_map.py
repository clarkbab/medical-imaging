import numpy as np
import os
import pandas as pd
import re

from mymi import config
from mymi import regions

class RegionMap:
    def __init__(
        self,
        dataset: pd.DataFrame):
        """
        args:
            map: the mapping data.
        """
        # Check for region map.
        filepath = os.path.join(config.directories.datasets, dataset, 'region-map.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"Region map doesn't exist, please create at '{filepath}'.")

        # Load map file.
        df = pd.read_csv(filepath)

        # Check that internal region names are entered correctly.
        for n in df.internal:
            if not regions.is_region(n):
                raise ValueError(f"Error in region map for dataset '{self._name}', '{n}' is not an internal region.")

        # Check that dataset region names are entered correctly. 
        # region_names = self.region_names(clear_cache=clear_cache).region.unique()
        # for n in df.dataset:
        #     if not n in region_names:
        #         raise ValueError(f"Error in region map for dataset '{self._name}', '{n}' is not a dataset region.")
        self._data = df

    @property
    def data(self) -> pd.DataFrame:
        return self._data

    def to_internal(
        self,
        region: str) -> str:
        """
        returns: the internal region name if appropriate mapping was supplied. If no mapping
            was supplied for the region then it remains unchanged.
        args:
            region: the region name to map.
        """
        # Iterrate over map rows.
        for _, row in self._data.iterrows():
            # Create pattern match args.
            args = [row.dataset, region]
            
            # Check case.
            case_matters = row['case-sensitive']
            if not np.isnan(case_matters) and not case_matters:
                args += [re.IGNORECASE]
                
            # Perform match.
            if re.match(*args):
                return row.internal
            
        return region
