import numpy as np
import os
from typing import *

from mymi.regions import regions_is_all, regions_to_list
from mymi.typing import *
from mymi.utils import *

class TrainingSample:
    def __init__(
        self,
        split: 'HoldoutSplit',
        id: SampleID) -> None:
        self.__split = split
        self.__id = int(id)     # Could be passed as a string by mistake.
        self.__index = None
        self.__global_id = f'{self.__split}:{self.__id}'

        # Define paths.
        self.__input_path = os.path.join(self.split.path, 'inputs', f"{self.__id:03}.npz")

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
        input = np.load(self.__input_path)['data']
        return input

    @property
    def input_path(self) -> str:
        return self.__input_path

    @property
    def origin(self) -> Tuple[str]:
        origin = [self.index['origin-dataset'], self.index['origin-patient-id']]
        opt_vals = ['origin-study-id', 'origin-fixed-study-id', 'origin-moving-study-id']
        for o in opt_vals:
            if o in self.index:
                origin.append(self.index[o])
        return tuple(origin)

    @property
    def size(self) -> ImageSize3D:
        return self.input.shape

    @property
    def spacing(self) -> ImageSpacing3D:
        return self.__split.dataset.spacing

    @property
    def split(self) -> 'HoldoutSplit':
        return self.__split

    # We have to filter by 'regions' here, otherwise we'd have to create a new dataset
    # for each different combination of 'regions' we want to train. This would create a
    # a lot of datasets for the multi-organ work.
    def label(
        self,
        landmarks: Landmarks = 'all',
        landmark_data_only: bool = True,
        label_idx: Optional[int] = None,    # Enables multi-label training.
        regions: Regions = 'all') -> np.ndarray:
        # Get label type.
        label_types = self.split.dataset.label_types
        if len(label_types) == 1:
            label_idx = 0
        elif label_idx is None:
            raise ValueError("Multiple labels present - must specify 'label_idx'.")
        label_type = label_types[label_idx]
        label_id = f'{self.__id:03}-{label_idx}' if len(label_types) > 1 else f'{self.__id:03}'  # Don't need suffix if single-label.

        if label_type == 'image':
            # Load image label.
            filepath = os.path.join(self.split.path, 'labels', f'{label_id}.npz')
            label = np.load(filepath)['data']

        elif label_type == 'regions':
            # Load regions label - slightly different to an 'image' label, as we need to 
            # check requested 'regions', and set channels accordingly.
            filepath = os.path.join(self.split.path, 'labels', f'{label_id}.npz')
            label = np.load(filepath)['data']
            if regions == 'all':
                return label

            # Filter regions.
            # Note: 'label' should return all 'regions' required for training, not just those 
            # present for this sample, as otherwise our label volumes will have different numbers
            # of channels between samples.
            all_regions = self.split.dataset.regions
            regions = regions_to_list(regions, literals={ 'all': all_regions })
            
            # Raise error if sample has no requested regions - the label will be full of zeros.
            if not self.has_regions(regions):
                raise ValueError(f"Sample {self.__id} has no regions {regions}.")

            # Extract requested 'regions'.
            channels = [0]
            channels += [all_regions.index(r) + 1 for r in regions]
            label = label[channels]

        elif label_type == 'landmarks':
            # Load landmarks dataframe.
            filepath = os.path.join(self.split.path, 'labels', f'{label_id}.csv')
            label = load_csv(filepath)
            if landmarks != 'all':
                # Filter on requested landmarks.
                landmarks = arg_to_list(landmarks, str, literals={ 'all': self.split.dataset.list_landmarks })
                label = label[label['landmark-id'].isin(landmarks)]
            label = label.rename(columns={ '0': 0, '1': 1, '2': 2 })

            if landmark_data_only:
                # Return coordinates only - tensors don't handle multiple data types.
                label = label[list(range(3))].to_numpy()

        return label

    def has_regions(
        self,
        regions: Regions,
        all: bool = False) -> bool:
        if regions_is_all(regions):
            return True

        regions = regions_to_list(regions)
        n_matching = len(np.intersect1d(regions, self.regions()))

        if n_matching == len(regions):
            return True
        elif not all and n_matching > 0:
            return True

        return False

    def mask(
        self,
        label_idx: Optional[int] = None,    # Enables multi-label training.
        regions: Regions = 'all') -> np.ndarray:
        label_types = self.split.dataset.label_types
        if len(label_types) == 1:
            label_idx = 0
        elif label_idx is None:
            raise ValueError("Multiple labels present - must specify 'label_idx'.")
        label_type = label_types[label_idx]
        if label_type != 'regions':
            raise ValueError(f"Mask only available for 'regions' labels, not '{label_type}'.")

        label_id = f'{self.__id:03}-{label_idx}' if len(label_types) > 1 else f'{self.__id:03}'  # Don't need suffix if single-label.
        filepath = os.path.join(self.split.path, 'masks', f"{label_id}.npz")
        mask = np.load(filepath)['data']
        if regions == 'all':
            return mask

        # Filter regions.
        # Note: 'mask' should return all 'regions' required for training, not just those 
        # present for this sample, as otherwise our masks will have different numbers
        # of channels between samples.
        all_regions = self.split.dataset.regions
        regions = regions_to_list(regions, literals={ 'all': all_regions })

        # Extract requested 'regions'.
        channels = [0]
        channels += [all_regions.index(r) + 1 for r in regions]
        mask = mask[channels]
        return mask

    def pair(
        self,
        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.input, self.label(**kwargs)

    def regions(
        self,
        label_idxs: Optional[Union[int, Sequence[int]]] = 'all') -> List[Region]:
        label_types = self.split.dataset.label_types
        if len(label_types) == 1:
            label_idxs = [0]
        else:
            def all_region_label_idxs() -> List[str]:
                return [i for i, l in enumerate(label_types) if l == 'regions']
            label_idxs = arg_to_list(label_idxs, int, literals={ 'all': all_region_label_idxs })
            if label_idxs is None:
                raise ValueError("Multiple labels present - must specify 'label_idxs'.")
            else:
                for i in label_idxs:
                    if label_types[i] != 'regions':
                        raise ValueError(f"Only 'regions' type label_idxs can be passed for sample 'regions'. Got '{i}', type '{label_types[i]}'.")
        label_types = [label_types[i] for i in label_idxs]
        all_regions = self.split.dataset.regions
        if all_regions is None:
            return None

        include = [False] * len(all_regions)
        for i in label_idxs:
            mask = self.mask(label_idx=i)[1:]
            for j, m in enumerate(mask):
                if m:
                    include[j] = True
        regions = [r for i, r in enumerate(all_regions) if include[i]]

        return regions

    def __str__(self) -> str:
        return self.__global_id
