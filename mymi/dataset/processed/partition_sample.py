from numpy.lib.arraysetops import intersect1d
from mymi.types.types import PatientRegions
import numpy as np
import os
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
        """
        returns: True if the patient has (at least) one of the requested regions.
        args:
            regions: the region names.
        """
        # Convert regions to list.
        all_regions = self.list_regions()
        regions = to_list(regions, all_regions)

        if len(np.intersect1d(regions, all_regions)) != 0:
            return True
        else:
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
