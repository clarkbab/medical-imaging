from itertools import chain
import numpy as np
from pytorch_lightning import seed_everything
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union

from mymi.types import ImageSpacing3D, PatientRegion, PatientRegions
from mymi import dataset as ds
from mymi.dataset.training import TrainingDataset
from mymi.geometry import get_centre
from mymi import logging
from torchio.transforms import Transform
from mymi.regions import regions_to_list
from mymi.transforms import centre_crop_or_pad_3D
from mymi.utils import arg_to_list

from .random_sampler import RandomSampler

def collate_fn(batch) -> List[Tensor]:
    # Get spatial dimensions of batch.
    max_size = [-np.inf, -np.inf, -np.inf]
    for _, input, _, _, _ in batch:     # Batch consists of (desc, input, label, mask, weights).
        size = input.shape[1:]
        for axis in range(3):
            if size[axis] > max_size[axis]:
                max_size[axis] = size[axis]
    max_size = tuple(max_size)

    # Gather all batch items.
    descs = []
    inputs = []
    labels = []
    masks = []
    weights = []
    for desc, input, label, mask, weight in batch:
        descs.append(desc)
        input_cs = []
        for c in range(len(input)):     # Perform pad separately for each channel as 'centre_crop_or_pad_4D' hasn't been written.
            input_c = centre_crop_or_pad_3D(input[c], max_size)
            input_cs.append(input_c)
        input = np.stack(input_cs, axis=0)
        inputs.append(input)
        label_cs = []
        for c in range(len(label)): 
            label_c = centre_crop_or_pad_3D(label[c], max_size)
            label_cs.append(label_c)
        label = np.stack(label_cs, axis=0)
        labels.append(label)
        masks.append(mask)
        weights.append(weight)

    # Stack batch items.
    desc = tuple(descs)
    input = np.stack(inputs, axis=0)
    label = np.stack(labels, axis=0)
    mask = np.stack(masks, axis=0)
    weights = np.stack(weights, axis=0)

    # Convert to pytorch tensors.
    input = torch.from_numpy(input)
    label = torch.from_numpy(label)
    mask = torch.from_numpy(mask)
    weights = torch.from_numpy(weights)

    return (desc, input, label, mask, weights)

class HoldoutLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        batch_size: int = 1,
        n_workers: int = 1,
        preload_data: bool = True,
        random_seed: int = 42,
        shuffle_train: bool = True,
        train_transform: Optional[Transform] = None,
        validate_transform: Optional[Transform] = None,
        **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:

        # Create train loader.
        col_fn = collate_fn if batch_size > 1 else None
        train_set = TrainingSet(dataset, transform=train_transform) 
        if shuffle_train:
            train_sampler = RandomSampler(train_set, random_seed=random_seed)
        else:
            train_sampler = None
        train_loader = DataLoader(batch_size=batch_size, collate_fn=col_fn, dataset=train_set, num_workers=n_workers, sampler=train_sampler, shuffle=False)

        # Create validate loader.
        val_ds = TrainingSet(dataset, regions, val_samples, class_weights=class_weights, data_hook=data_hook, include_background=include_background, load_data=load_data, load_origin=load_train_origin, preload_data=preload_data, spacing=spacing, transform=transform_val)
        val_loader = DataLoader(batch_size=batch_size, collate_fn=col_fn, dataset=val_ds, num_workers=n_workers, shuffle=False)

        # Create test loader.
        if n_folds is not None or use_split_file:
            test_ds = TestSet(datasets, test_samples, load_origin=load_test_origin) 
            test_loader = DataLoader(batch_size=batch_size, dataset=test_ds, num_workers=n_workers, shuffle=False)

        # Create subtest loader.
        if (n_folds is not None or use_split_file) and n_subfolds is not None:
            test_ds = TestSet(datasets, test_subsamples, load_origin=load_test_origin) 
            subtest_loader = DataLoader(batch_size=batch_size, dataset=test_ds, num_workers=n_workers, shuffle=False)

        # Return loaders.
        if n_folds is not None or use_split_file:
            if n_subfolds is not None:
                return train_loader, val_loader, subtest_loader, test_loader 
            else:
                return train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader

class TrainingSet(Dataset):
    def __init__(
        self,
        dataset: str,
        split: str,
        spacing: Optional[ImageSpacing3D] = None,
        transform: torchio.transforms.Transform = None):
        self.__class_weights = class_weights
        self.__data_hook = data_hook
        self.__include_background = include_background
        self.__load_data = load_data
        self.__load_origin = load_origin
        self.__preload_data = preload_data
        self.__random_seed = random_seed
        self.__regions = regions
        self.__spacing = spacing
        self.__transform = transform
        if transform:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'
        
        # Load datasets.
        self.__sets = [ds.get(dataset, 'training') for dataset in datasets]

        # Record number of samples.
        self.__n_samples = len(samples)

        # Map loader indices to dataset indices.
        self.__sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

        # Create map from region names to channels.
        self.__n_channels = len(self.__regions) + 1
        self.__region_channel_map = { 'background': 0 }
        for i, region in enumerate(self.__regions):
            self.__region_channel_map[region] = i + 1

        # Calculate region counts.
        region_counts = np.zeros(self.__n_channels, dtype=int)
        for ds_i, s_i in samples:
            regions = self.__sets[ds_i].sample(s_i).list_regions()
            regions = [r for r in regions if r in self.__regions]
            for region in regions:
                region_counts[self.__region_channel_map[region]] += 1

            # If all regions are present, we can train background class.
            if self.__include_background:
                if len(regions) == len(self.__regions):
                    region_counts[0] += 1

        logging.info(f"Calculated region counts '{region_counts}'.")

        if class_weights is None:
            # If class has no labels, set weight=0.
            class_weights = []
            for i in range(len(region_counts)):
                if region_counts[i] == 0:
                    class_weights.append(0)
                else:
                    class_weights.append(1 / region_counts[i])

            # Normalise weight values.
            class_weights = class_weights / np.sum(class_weights)

        logging.info(f"Using class weights '{class_weights}'.")
        self.__class_weights = class_weights

        # Count iterations for reproducibility.
        self.__n_iter = 0

        # Preload data.
        if self.__preload_data:
            logging.info(f"Preloading data for {self.__n_samples} samples.")
            self.__data = []
            for i in tqdm(range(self.__n_samples)):
                # Get dataset sample.
                ds_i, s_i = self.__sample_map[i]
                set = self.__sets[ds_i]

                # Load region data.
                sample = set.sample(s_i)
                regions = sample.list_regions(regions=self.__regions)
                input, labels = sample.pair(region=regions)
                self.__data.append((input, labels))

    def __len__(self):
        return self.__n_samples

    def __getitem__(
        self,
        index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get dataset/sample.
        ds_i, s_i = self.__sample_map[index]
        set = self.__sets[ds_i]

        # Get description.
        desc = f'{set.name}:{s_i}'
        if not self.__load_data:
            if self.__load_origin:
                desc = ':'.join((str(el) for el in set.sample(s_i).origin))
            return desc

        # Get sample regions.
        sample = set.sample(s_i)
        regions = sample.list_regions(regions=self.__regions)

        # Load input/labels.
        if self.__preload_data:
            input, labels = self.__data[index]
        else:
            input, labels = sample.pair(region=regions)

        # Apply data hook.
        if self.__data_hook is not None:
            input, labels = self.__data_hook(set.name, s_i, input, labels, spacing=self.__spacing)

        # Create multi-class mask and label.
        # Note that using this method we may end up with multiple foreground classes for a
        # single voxel. E.g. brain/brainstem both present. Don't worry about this for now,
        # the network will just try to maximise both (and fail).
        mask = np.zeros(self.__n_channels, dtype=bool)
        label = np.zeros((self.__n_channels, *input.shape), dtype=bool)
        for region in regions:
            mask[self.__region_channel_map[region]] = True
            label[self.__region_channel_map[region]] = labels[region]

        # Add background class.
        # When all foreground regions are annotated, we can invert their union to return background label.
        # If a region is missing, we can't get the background label as we don't know which voxels are foreground
        # for the missing region and which are background.
        if self.__include_background and len(regions) == len(self.__regions):
            mask[0] = True
            label[0] = np.invert(label.any(axis=0))

        # Perform transform.
        if self.__transform:
            # Transform input/labels.
            affine = np.array([
                [self.__spacing[0], 0, 0, 0],
                [0, self.__spacing[1], 0, 0],
                [0, 0, self.__spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = np.expand_dims(input, axis=0)
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject = Subject({
                'input': input,
                'label': label
            })

            # Transform the subject.
            # seed = self.__random_seed + index
            # print(f"(pid={os.getpid()},index={index}) seeding transform {seed}")
            # seed_everything(seed)   # Ensure reproducibility when resuming training.
            output = self.__transform(subject)

            # Remove 'channel' dimension.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

            # Convert to numpy.
            input = input.numpy()
            label = label.numpy().astype(bool)

        # Add channel dimension - expected by pytorch.
        input = np.expand_dims(input, 0)

        # Increment counter.
        self.__n_iter += 1

        return desc, input, label, mask, self.__class_weights
    
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
