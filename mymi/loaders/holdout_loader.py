import itertools
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio.transforms import Transform
from tqdm import tqdm
from typing import *

from mymi import datasets as ds
from mymi.datasets.training import TrainingDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import resample, sitk_transform_points
from mymi.typing import *
from mymi.utils import *

from .random_sampler import RandomSampler

class HoldoutLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        batch_size: int = 1,
        landmarks: Optional[Landmarks] = 'all',
        normalise: bool = True,
        normalise_by_channel: bool = False,
        norm_params: Optional[Dict[str, float]] = None,
        n_workers: int = 1,
        pad_fill: Optional[float] = -1024,
        pad_threshold: Optional[float] = -1024,
        preload_samples: bool = True,
        regions: Optional[Regions] = 'all',
        shuffle_train: bool = True,
        transform_train: Optional[Transform] = None,
        **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, float]]:
        set = TrainingDataset(dataset)
        regions = regions_to_list(regions, literals={ 'all': set.regions })

        # Create train loader.
        train_split = set.split('train')
        okwargs = dict(
            landmarks=landmarks,
            normalise=normalise,
            normalise_by_channel=normalise_by_channel,
            norm_params=norm_params,
            pad_fill=pad_fill,
            pad_threshold=pad_threshold,
            preload_samples=preload_samples,
            regions=regions,
            spacing=set.spacing,
            transform=transform_train,
        )
        train_set = TrainingSet(train_split, **okwargs) 
        train_sampler = RandomSampler(train_set) if shuffle_train else None
        train_loader = DataLoader(batch_size=batch_size, dataset=train_set, num_workers=n_workers, sampler=train_sampler, shuffle=False)

        # Create validate loader.
        val_split = set.split('validate')
        okwargs = dict(
            landmarks=landmarks,
            normalise=normalise,
            normalise_by_channel=normalise_by_channel,
            norm_params=train_set.norm_params,
            pad_fill=pad_fill,
            pad_threshold=pad_threshold,
            preload_samples=preload_samples,
            regions=regions,
        ) 
        val_set = TrainingSet(val_split, **okwargs)
        val_loader = DataLoader(batch_size=batch_size, dataset=val_set, num_workers=n_workers, shuffle=False)

        return train_loader, val_loader, None, train_set.norm_params

class TrainingSet(Dataset):
    def __init__(
        self,
        split: 'HoldoutSplit',
        landmarks: Optional[Landmarks] = 'all',
        normalise: bool = True,
        normalise_by_channel: bool = False,
        norm_params: Optional[Dict[str, float]] = None,
        pad_fill: Union[float, Literal['min']] = 'min',
        pad_threshold: Optional[float] = None,
        preload_samples: bool = True,
        regions: Optional[Regions] = 'all',
        spacing: Optional[Spacing3D] = None,
        transform: torchio.transforms.Transform = None) -> None:
        self.__landmarks = landmarks
        self.__label_types = split.dataset.label_types
        self.__normalise = normalise
        self.__normalise_by_channel = normalise_by_channel
        self.__norm_params = norm_params
        self.__pad_fill = pad_fill
        self.__pad_threshold = pad_threshold
        self.__preload_samples = preload_samples
        self.__regions = regions
        self.__spacing = spacing
        self.__split = split
        self.__transform = transform
        if self.__transform is not None:
            assert self.__spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Record number of samples.
        self.__sample_ids = self.__split.list_samples(regions=self.__regions)
        self.__n_samples = len(self.__sample_ids)

        if self.__normalise:
            # Handle input with/without channels.
            input_shape = self.__split.sample(self.__sample_ids[0]).input.shape
            if len(input_shape) == 3:
                self.__n_channels = None
                self.__normalise_by_channel = False
            elif len(input_shape) == 4:
                self.__n_channels = input_shape[0]
            else:
                raise ValueError(f"Unrecognised input shape '{input_shape}'.")

            if self.__norm_params is None:
                logging.info("Calculating normalisation parameters...")

                # Calculate normalisation parameters per-channel (in case of multi-modal input).
                if self.__normalise_by_channel:
                    datas = dict((c, []) for c in range(self.__n_channels))
                else:
                    datas = []
                for s in self.__sample_ids:
                    input = self.__split.sample(s).input
                    if self.__normalise_by_channel:
                        for c in range(self.__n_channels):
                            data = input[c].flatten()
                            # Don't include padding values for normalisation.
                            if self.__pad_threshold is not None:
                                data = data[data >= self.__pad_threshold]
                            datas[c].append(data)
                    else:
                        data = input.flatten()
                        # Don't include padding values for normalisation.
                        if self.__pad_threshold is not None:
                            data = data[data >= self.__pad_threshold]
                        datas.append(data)

                if self.__normalise_by_channel:
                    self.__norm_params = {}
                    for c in range(self.__n_channels):
                        self.__norm_params[c] = { 'mean': np.mean(np.concatenate(datas[c])), 'std': np.std(np.concatenate(datas[c])) }
                else:
                    self.__norm_params = { 'mean': np.mean(np.concatenate(datas)), 'std': np.std(np.concatenate(datas)) }
        
            logging.info(f"Applying normalisation using parameters: {self.__norm_params}.")

        # Preload samples.
        if self.__preload_samples:
            logging.info(f"Preloading training samples (n={self.__n_samples}).")
            self.__inputs = []
            self.__labels = dict((i, []) for i in range(len(self.__sample_ids)))    # Enable multi-label training.
            self.__masks = dict((i, []) for i in range(len(self.__sample_ids)))    # Enable multi-label training.
            for i, s in tqdm(enumerate(self.__sample_ids)):
                # Load input.
                sample = self.__split.sample(s)
                input = sample.input

                # Replace padding if necessary.
                if self.__pad_threshold is not None:
                    fill = self.__pad_fill if self.__pad_fill != 'min' else np.min(input[input >= self.__pad_threshold])
                    input[input < self.__pad_threshold] = fill
                self.__inputs.append(input)

                # Add preloaded labels and masks (multiple allowed per sample, e.g. labels and landmarks).
                for j, l in enumerate(self.__label_types):
                    # Load label.
                    okwargs = dict(
                        label_idx=j,
                    )
                    if l == 'regions':
                        okwargs['regions'] = self.__regions
                    elif l == 'landmarks':
                        okwargs['landmarks'] = self.__landmarks
                    label = sample.label(**okwargs)
                    self.__labels[i].append(label)

                    if self.__regions is not None and l == 'regions':
                        mask = sample.mask(label_idx=j, regions=self.__regions)
                        self.__masks[i].append(mask)

    def __len__(self):
        return self.__n_samples

    @property
    def norm_params(self) -> Optional[Dict[str, float]]:
        return self.__norm_params

    def __getitem__(
        self,
        idx: int) -> Tuple[Any]:
        # Create description:
        sample_id = self.__sample_ids[idx]
        sample = self.__split.sample(sample_id)
        desc = str(sample)

        # Load data.
        if self.__preload_samples:
            input = self.__inputs[idx]
            labels = self.__labels[idx]     #  Could be multiple labels.
            if self.__regions is not None:
                masks = self.__masks[idx]   # Could be multiple masks.
        else:
            # Load input.
            input = sample.input

            # Replace padding values.
            if self.__pad_threshold is not None:
                fill = self.__pad_fill if self.__pad_fill != 'min' else np.min(input[input >= self.__pad_threshold])
                input[input < self.__pad_threshold] = fill

            # Load multiple outputs.
            labels = []
            masks = []
            for i, l in enumerate(self.__label_types):
                # Load label.
                okwargs = dict(
                    label_idx=i,
                )
                if l == 'regions':
                    okwargs['regions'] = self.__regions
                elif l == 'landmarks':
                    okwargs['landmarks'] = self.__landmarks
                label = sample.label(**okwargs)
                labels.append(label)

                if self.__regions is not None:
                    mask = sample.mask(label_idx=j, regions=self.__regions)
                    masks.append(mask)

        if self.__transform is not None:
            # Get concrete transform.
            origin = (0, 0, 0)
            transform_f, transform_b, _ = self.__transform.get_concrete_transform()

            # Transform input.
            input = resample(input, fill=self.__pad_fill, spacing=self.__spacing, transform=transform_b)

            # Transform labels.
            for i, l in enumerate(self.__label_types):
                if l in ('image', 'regions'):
                    labels[i] = resample(labels[i], fill=0, spacing=self.__spacing, transform=transform_b)
                elif l == 'landmarks':
                    label = sitk_transform_points(labels[i], transform_f)

                    # For landmarks that are transformed outside of the view window, replace with nans.
                    # We can't remove entirely as we'd lose our implicit 'landmark-id' that we obtain from the
                    # first array axis.
                    max_point = np.array(self.__spacing) * spatial_shape
                    boundary = np.array([origin, max_point], dtype=np.float64)
                    boundary_t = sitk_transform_points(boundary, transform_f)
                    for j, point in enumerate(label):
                        for a, pa in enumerate(point):
                            if pa < boundary_t[0][a] or pa > boundary_t[1][a]:
                                label[j] = np.full(3, np.nan)

                    labels[i] = label

        # Add channel dimension - expected by pytorch.
        if len(input.shape) == 3:
            input = np.expand_dims(input, 0)
        for i, l in enumerate(self.__label_types):
            # Images are usually stored as no-channel volumes.
            if l == 'image' and len(labels[i].shape) == 3:
                labels[i] = np.expand_dims(labels[i], 0)

        # Normalise input data.
        if self.__normalise:
            if self.__normalise_by_channel:
                for c in range(self.__n_channels):
                    input[c] = (input[c] - self.__norm_params[c]['mean']) / self.__norm_params[c]['std']
            else:
                input = (input - self.__norm_params['mean']) / self.__norm_params['std']

        # Cast to required training types.
        input = input.astype(np.float32)

        # Handle return values. Can't pass None, as PyTorch doesn't recognise this
        # as a type.
        return_vals = [desc, input, labels]
        if self.__regions is not None:
            return_vals += masks

        return tuple(return_vals)
    
class TestSet(Dataset):
    def __init__(
        self,
        datasets: List[str],
        samples: List[Tuple[int, int]],
        load_origin: bool = True):
        self.__sets = [ds.get(dataset, 'training') for dataset in datasets]
        self.__load_origin = load_origin

        # Record number of samples.
        self.__n_samples = len(samples)

        # Map loader indices to dataset indices.
        self.__sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

    def __len__(self):
        return self.__n_samples

    def __getitem__(
        self,
        index: int) -> Tuple[str]:
        # Load data.
        ds_i, s_i = self.__sample_map[index]
        set = self.__sets[ds_i]
        
        if self.__load_origin:
            # Return 'NIFTI' location of training sample.
            desc = ':'.join((str(el) for el in set.sample(s_i).origin))
        else:
            desc = f'{set.name}:{s_i}'

        return desc
