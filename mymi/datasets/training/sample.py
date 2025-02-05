import numpy as np
import os
from typing import *

from mymi.regions import regions_to_list
from mymi.typing import *

class TrainingSample:
    def __init__(
        self,
        split: 'TrainingSplit',
        id: int) -> None:
        self.__split = split
        self.__id = id
        self.__index = None
        self.__global_id = f'{self.__split}:{self.__id}'

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> str:
        if self.__index is None:
            s_index = self.split.index
            self.__index = s_index[s_index['sample-id'] == self.__id].iloc[0].copy()
        return self.__index

    @property
    def input(self) -> np.ndarray:
        filepath = os.path.join(self.split.path, 'inputs', f"{self.__id:03}.npz")
        input = np.load(filepath)['data']
        return input

    @property
    def origin(self) -> Tuple[str, str, str]:
        return self.index['origin-dataset'], self.index['origin-patient-id'], self.index['origin-study-id']

    @property
    def size(self) -> ImageSize3D:
        return self.input.shape

    @property
    def spacing(self) -> ImageSpacing3D:
        return self.__split.dataset.spacing

    @property
    def split(self) -> 'TrainingSplit':
        return self.__split

    # We have to filter by 'regions' here, otherwise we'd have to create a new dataset
    # for each different combination of 'regions' we want to train. This would create a
    # a lot of datasets for the multi-organ work.
    def label(
        self,
        regions: PatientRegions = 'all') -> np.ndarray:
        filepath = os.path.join(self.split.path, 'labels', f"{self.__id:03}.npz")
        label = np.load(filepath)['data']
        if regions == 'all':
            return label

        # Filter regions.
        # Note: 'label' should return all 'regions' required for training, not just those 
        # present for this sample, as otherwise our label volumes will have different numbers
        # of channels between samples.
        regions = regions_to_list(regions)

        # Extract requested 'regions'.
        all_regions = self.split.dataset.list_regions()
        channels = [0]
        channels += [all_regions.index(r) + 1 for r in regions]
        label = label[channels]

        return label

    def list_regions(self) -> List[PatientRegion]:
        all_regions = self.split.dataset.list_regions()
        regions = [r for r, m in zip(all_regions, self.mask()[1:]) if m]
        return regions

    def mask(
        self,
        regions: PatientRegions = 'all') -> np.ndarray:
        filepath = os.path.join(self.split.path, 'masks', f"{self.__id:03}.npz")
        mask = np.load(filepath)['data']
        if regions == 'all':
            return mask

        # Filter regions.
        # Note: 'mask' should return all 'regions' required for training, not just those 
        # present for this sample, as otherwise our masks will have different numbers
        # of channels between samples.
        regions = regions_to_list(regions)

        # Extract requested 'regions'.
        all_regions = self.split.dataset.list_regions()
        channels = [0]
        channels += [all_regions.index(r) + 1 for r in regions]
        mask = mask[channels]
        return mask

    def pair(
        self,
        regions: PatientRegions = 'all') -> Tuple[np.ndarray, np.ndarray]:
        return self.input, self.label(regions=regions)

    def __str__(self) -> str:
        return self.__global_id
