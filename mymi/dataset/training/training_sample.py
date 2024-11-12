from numpy.lib.arraysetops import intersect1d
from mymi.types import ImageSize3D, ImageSpacing3D, PatientRegions
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from mymi.regions import regions_to_list
from mymi.types import PatientRegions

class TrainingSample:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        id: Union[int, str]):
        self.__dataset = dataset
        self.__id = int(id)
        self.__index = None         # Lazy-loaded.
        self.__global_id = f'{self.__dataset} - {self.__id}'
        self.__group = None      # Lazy-loaded.
        self.__spacing = self.__dataset.params['spacing'] if 'spacing' in self.__dataset.params else self.__dataset.params['output-spacing']    

        # Load sample index.
        if self.__id not in self.__dataset.list_samples():
            raise ValueError(f"Sample '{self.__id}' not found for dataset '{self.__dataset}'.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def group(self) -> str:
        if self.__group is None:
            if 'group-id' in self.index.columns:
                self.__group = self.index.iloc[0]['group-id']
            else:
                self.__group = None
        return self.__group

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> str:
        if self.__index is None:
            self.__load_index()
        return self.__index

    @property
    def input(self) -> np.ndarray:
        # Load the input data.
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}.npz')
        if not os.path.exists(filepath):
            raise ValueError(f"Input data not found for sample '{self}'.")
        data = np.load(filepath)['data']
        data = data.astype(np.float32)
        return data

    @property
    def origin(self) -> Tuple:
        idx = self.__dataset.index
        record = idx[idx['sample-id'] == self.__id].iloc[0]
        return (record['origin-dataset'], record['origin-patient-id'])

    @property
    def size(self) -> ImageSize3D:
        return self.input.shape

    @property
    def spacing(self) -> ImageSpacing3D:
        return self.__spacing

    def list_regions(
        self,
        include_empty: bool = False,
        only: Optional[PatientRegions] = None) -> List[str]:
        # Filter empty region.
        df = self.index
        if 'empty' in df and not include_empty:
            df = df[~df['empty']]
        regions = list(sorted(df['region']))

        # Filter on 'only'.
        only = regions_to_list(only)
        if only is not None:
            regions = [r for r in regions if r in only]

        return regions

    def has_region(
        self,
        regions: PatientRegions) -> bool:
        regions = regions_to_list(regions)
        pat_regions = self.list_regions()
        for region in regions:
            if region in pat_regions:
                return True
        return False

    def label(
        self,
        regions: PatientRegions = 'all',
        regions_ignore_missing: bool = False) -> Dict[str, np.ndarray]:
        regions = regions_to_list(regions, literals={ 'all': self.list_regions() })

        # Load the label data.
        data = {}
        for region in regions:
            filepath = os.path.join(self.__dataset.path, 'data', 'labels', region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                if regions_ignore_missing:
                    continue
                else:
                    raise ValueError(f"Label '{region}' not found for sample '{self}'.")
            label = np.load(filepath)['data']
            data[region] = label

        return data

    def pair(
        self,
        regions: PatientRegions = 'all') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.input, self.label(regions=regions)

    def __load_index(self) -> None:
        index = self.__dataset.index
        index = index[index['sample-id'] == self.__id]
        assert len(index == 1)
        self.__index = index

    def __str__(self) -> str:
        return self.__global_id
