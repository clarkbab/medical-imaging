from numpy.lib.arraysetops import intersect1d
from mymi.types.types import PatientRegions
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple

from mymi.regions import to_list
from mymi import types

class TrainingSample:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        index: int):
        if index not in dataset.list_samples():
            raise ValueError(f"Sample '{index}' not found for dataset '{dataset}'.")
        self._global_id = f'{dataset} - {index}'
        self._dataset = dataset
        self._index = index
        self._spacing = dataset.params['spacing']

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def index(self) -> str:
        return self._index

    @property
    def spacing(self) -> types.ImageSpacing3D:
        return self._spacing

    def list_regions(self) -> List[str]:
        """
        returns: the region names.
        """
        # List all regions.
        filepath = os.path.join(self._dataset.path, 'data', 'labels')
        all_regions = os.listdir(filepath)

        def filter_fn(region):
            filepath = os.path.join(self._dataset._path, 'data', 'labels', region, f'{self._index}.npz')
            if os.path.exists(filepath):
                return True
            else:
                return False
        return list(sorted(filter(filter_fn, all_regions)))

    def has_region(
        self,
        region: str) -> bool:
        return region in self.list_regions()

    @property
    def origin(self) -> Tuple[str, str]:
        manifest = self._dataset.manifest
        record = manifest[manifest['index'] == self._index].iloc[0]
        return record['dataset'], record['patient-id']

    @property
    def input(self) -> np.ndarray:
        # Load the input data.
        filepath = os.path.join(self._dataset.path, 'data', 'inputs', f'{self._index}.npz')
        data = np.load(filepath)['data']
        return data

    def label(
        self,
        regions: types.PatientRegions = 'all') -> Dict[str, np.ndarray]:
        # Convert regions to list.
        if type(regions) == str:
            if regions == 'all':
                regions = list(sorted(self.list_regions))
            else:
                regions = [regions]

        # Load the label data.
        data = {}
        for region in regions:
            filepath = os.path.join(self._dataset._path, 'data', 'labels', region, f'{self._index}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Region '{region}' not found for sample '{self}'.")
            label = np.load(filepath)['data']
            data[region] = label

        return data

    def pair(
        self,
        regions: types.PatientRegions = 'all') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.input, self.label(regions)
