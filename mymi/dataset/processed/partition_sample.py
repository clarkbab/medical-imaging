from numpy.lib.arraysetops import intersect1d
from mymi.types.types import PatientRegions
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple

from mymi.regions import to_list
from mymi import types

FILENAME_NUM_DIGITS = 5

class PartitionSample:
    def __init__(
        self,
        partition: 'ProcessedPartition',
        index: int):
        self._partition = partition
        self._index = index

    def list_regions(self) -> List[str]:
        """
        returns: the region names.
        """
        regions = self._partition.list_regions()
        def filter_fn(region):
            filename = f"{self._index:0{FILENAME_NUM_DIGITS}}.npz"
            filepath = os.path.join(self._partition._path, 'labels', region, filename)
            if os.path.exists(filepath):
                return True
            else:
                return False
        return list(filter(filter_fn, regions))

    def has_one_region(
        self,
        regions: types.PatientRegions) -> bool:
        pat_regions = self.list_regions()
        if type(regions) == str:
            if regions == 'all' and len(pat_regions) != 0:
                return True
            elif regions in pat_regions:
                return True
            else:
                return False
        else:
            for region in regions:
                if region in pat_regions:
                    return True
            return False

    def input(self) -> np.ndarray:
        """
        returns: the input data for sample i.
        args:
            index: the sample index to load.
        """
        # Load the input data.
        filename = f"{self._index:0{FILENAME_NUM_DIGITS}}.npz"
        filepath = os.path.join(self._partition._path, 'inputs', filename)
        data = np.load(filepath)['data']
        return data

    def label(
        self,
        regions: types.PatientRegions = 'all') -> Dict[str, np.ndarray]:
        """
        returns: the label data for sample i.
        args:
            index: the sample index to load.
            regions: the regions to return.
        """
        # Convert regions to list.
        regions = to_list(regions, self.list_regions())
    
        # Load the label data.
        data = {}
        for region in regions:
            filename = f"{self._index:0{FILENAME_NUM_DIGITS}}.npz"
            filepath = os.path.join(self._partition._path, 'labels', region, filename)
            label = np.load(filepath)['data']
            data[region] = label
        return data

    def pair(
        self,
        regions: types.PatientRegions = 'all') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        returns: the (input, label) pair for the given index.
        kwargs:
            regions: the region to return.
        """
        return self.input(), self.label(regions=regions)

    def input_summary(self) -> pd.DataFrame:
        cols = {
            'size-x': int,
            'size-y': int,
            'size-z': int
        }
        df = pd.DataFrame(columns=cols.keys())
        input = self.input()
        data = {
            'size-x': input.shape[0],
            'size-y': input.shape[1],
            'size-z': input.shape[2]
        }
        df = df.append(data, ignore_index=True)
        df = df.astype(cols)
        return df

    def label_summary(
        self,
        regions: types.PatientRegions = 'all') -> pd.DataFrame:
        cols = {
            'region': str,
            'size-x': int,
            'size-y': int,
            'size-z': int
        }
        df = pd.DataFrame(columns=cols.keys())
        label = self.label(regions=regions)
        for region, ldata in label.items():
            data = {
                'region': region,
                'size-x': ldata.shape[0],
                'size-y': ldata.shape[1],
                'size-z': ldata.shape[2]
            }
            df = df.append(data, ignore_index=True)
        df = df.astype(cols)
        return df
