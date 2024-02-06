from numpy.lib.arraysetops import intersect1d
from mymi.types import ImageSize3D, ImageSpacing3D, PatientRegions
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from mymi import logging
from mymi.regions import region_to_list
from mymi import types
from mymi.utils import arg_to_list

class TrainingAdaptiveSample:
    def __init__(
        self,
        dataset: 'TrainingAdaptiveDataset',
        id: Union[int, str]):
        self.__dataset = dataset
        self.__id = int(id)
        self.__index = None         # Lazy-loaded.
        self.__global_id = f'{self.__dataset} - {self.__id}'
        self.__group_id = None      # Lazy-loaded.
        self.__spacing = self.__dataset.params['spacing']

        # Load sample index.
        if self.__id not in self.__dataset.list_samples():
            raise ValueError(f"Sample '{self.__id}' not found for dataset '{self.__dataset}'.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def group_id(self) -> str:
        if self.__group_id is None:
            self.__group_id = self.index.iloc[0]['group-id']
        return self.__group_id

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> str:
        if self.__index is None:
            self.__load_index()
        return self.__index

    @property
    def fixed_input(self) -> np.ndarray:
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}-1.npz')
        if not os.path.exists(filepath):
            raise ValueError(f"'fixed_input' not found for sample '{self}'. Filepath: '{filepath}'.")
        input = np.load(filepath)['data']
        return input

    @property
    def moving_input(self) -> np.ndarray:
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}-0.npz')
        if not os.path.exists(filepath):
            raise ValueError(f"'moving_input' not found for sample '{self}'. Filepath: '{filepath}'.")
        input = np.load(filepath)['data']
        return input

    @property
    def fixed_label(self) -> np.ndarray:
        label = None
        n_channels = len(self.__dataset.list_regions()) + 1
        for i, region in enumerate(self.__dataset.list_regions()):
            if not self.has_region(region):
                continue

            # Load the label data.
            filepath = os.path.join(self.__dataset.path, 'data', 'labels', region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Label (region={region}) not found for sample '{self}'. Filepath: '{filepath}'.")
            region_label = np.load(filepath)['data']
            if label is None:
                label = np.zeros((n_channels, *region_label.shape), dtype=np.bool_)
            label[i + 1] = region_label

        return label

    @property
    def moving_label(self) -> np.ndarray:
        label = None
        n_channels = len(self.__dataset.list_regions()) + 1
        for i, region in enumerate(self.__dataset.list_regions()):
            if not self.has_input_region(region):
                continue

            # Load the label data.
            filepath = os.path.join(self.__dataset.path, 'data', 'inputs', region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Label (region={region}) not found for sample '{self}'. Filepath: '{filepath}'.")
            region_label = np.load(filepath)['data']
            if label is None:
                label = np.zeros((n_channels, *region_label.shape), dtype=np.bool_)
            label[i + 1] = region_label

        return label

    @property
    def input(self) -> np.ndarray:
        # Load first 2 channels.
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}-0.npz')
        if not os.path.exists(filepath):
            raise ValueError(f"Input (channel=0) data not found for sample '{self}'. Filepath: '{filepath}'.")
        input_0 = np.load(filepath)['data']
        filepath = os.path.join(self.__dataset.path, 'data', 'inputs', f'{self.__id}-1.npz')
        if not os.path.exists(filepath):
            raise ValueError(f"Input (channel=1) data not found for sample '{self}'. Filepath: '{filepath}'.")
        input_1 = np.load(filepath)['data']
        
        # Create input holder.
        all_regions = self.__dataset.list_regions()
        n_channels = len(all_regions) + 2
        input = np.zeros((n_channels, *input_0.shape), dtype=np.float32)
        input[0] = input_0
        input[1] = input_1

        # Get other channels.
        folderpath = os.path.join(self.__dataset.path, 'data', 'inputs')
        for i, region in enumerate(all_regions):
            if not self.has_input_region(region):
                continue

            # Load channel data.
            filepath = os.path.join(folderpath, region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Input (region={region}) data not found for sample '{self}'. Filepath: '{filepath}'.")
            input_region = np.load(filepath)['data']
            input[i + 2] = input_region

        return input

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

    def list_input_regions(
        self,
        only: Optional[PatientRegions] = None) -> List[str]:
        regions = eval(self.index.iloc[0]['input-regions'])

        # Filter on 'only'.
        if only is not None:
            only = arg_to_list(only, str)
            regions = [r for r in regions if r in only]

        return regions

    def list_regions(
        self,
        only: Optional[PatientRegions] = None) -> List[str]:
        regions = eval(self.index.iloc[0]['regions'])

        # Filter on 'only'.
        if only is not None:
            only = arg_to_list(only, str)
            regions = [r for r in regions if r in only]

        return regions

    def has_input_region(
        self,
        region: PatientRegions) -> bool:
        regions = arg_to_list(region, str)
        pat_regions = self.list_input_regions()
        for region in regions:
            if region in pat_regions:
                return True
        return False

    def has_region(
        self,
        region: PatientRegions) -> bool:
        regions = arg_to_list(region, str)
        pat_regions = self.list_regions()
        for region in regions:
            if region in pat_regions:
                return True
        return False

    @property
    def label(self) -> np.ndarray:
        label = None
        n_channels = len(self.__dataset.list_regions()) + 1
        for i, region in enumerate(self.__dataset.list_regions()):
            if not self.has_region(region):
                continue

            # Load the label data.
            filepath = os.path.join(self.__dataset.path, 'data', 'labels', region, f'{self.__id}.npz')
            if not os.path.exists(filepath):
                raise ValueError(f"Label (region={region}) not found for sample '{self}'. Filepath: '{filepath}'.")
            region_label = np.load(filepath)['data']
            if label is None:
                label = np.zeros((n_channels, *region_label.shape), dtype=np.bool_)
            label[i + 1] = region_label

        return label

    @property
    def pair(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.input, self.label

    def __load_index(self) -> None:
        index = self.__dataset.index
        index = index[index['sample-id'] == self.__id]
        assert len(index == 1)
        self.__index = index

    def __str__(self) -> str:
        return self.__global_id
