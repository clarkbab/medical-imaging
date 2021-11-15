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
        partition: 'TrainingPartition',
        index: int):
        if index not in partition.list_samples():
            raise ValueError(f"Sample '{index}' not found for partition '{partition.name}', dataset '{partition.dataset.description}'.")
        self._global_id = f'{partition} - {index}'
        self._partition = partition
        self._index = index

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def patient_id(self) -> str:
        manifest_df = self._partition.dataset.manifest()
        pat_id = manifest_df[(manifest_df['partition'] == self._partition.name) & (manifest_df['index'] == self._index)].iloc[0]['patient-id']
        return pat_id

    def list_regions(self) -> List[str]:
        """
        returns: the region names.
        """
        # List all regions.
        filepath = os.path.join(self._partition.path, 'labels')
        all_regions = os.listdir(filepath)

        def filter_fn(region):
            filepath = os.path.join(self._partition._path, 'labels', region, f'{self._index}.npz')
            if os.path.exists(filepath):
                return True
            else:
                return False
        return list(filter(filter_fn, all_regions))

    def has_region(
        self,
        region: str) -> bool:
        return region in self.list_regions()

    def input(self) -> np.ndarray:
        """
        returns: the input data for sample i.
        args:
            index: the sample index to load.
        """
        # Load the input data.
        filepath = os.path.join(self._partition._path, 'inputs', f'{self._index}.npz')
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
        if type(regions) == str:
            if regions == 'all':
                regions = list(sorted(self.list_regions))
            else:
                regions = [regions]
    
        # Load the label data.
        data = {}
        for region in regions:
            filepath = os.path.join(self._partition._path, 'labels', region, f'{self._index}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Region '{region}' not found for sample '{self._index}', partition '{self._partition.name}', dataset '{self._partition._dataset.description}'.")
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
