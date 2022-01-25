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

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def index(self) -> str:
        return self._index

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
        return list(filter(filter_fn, all_regions))

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

    @property
    def label(self) -> Dict[str, np.ndarray]:
        # Load the label data.
        filepath = os.path.join(self._dataset.path, 'data', 'labels', f'{self._index}.npz')
        if not os.path.exists(filepath):
            raise ValueError(f"Label not found for sample '{self._index}', dataset '{self._dataset}'.")
        label = np.load(filepath)['data']
        return label

    @property
    def pair(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.input, self.label
