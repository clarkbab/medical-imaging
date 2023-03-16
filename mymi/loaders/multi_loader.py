import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Callable, List, Optional, Tuple, Union

from mymi.types import ImageSpacing3D, PatientRegions
from mymi import dataset as ds
from mymi.dataset.training import TrainingDataset
from mymi.geometry import get_centre
from mymi import logging
from mymi.regions import region_to_list
from mymi.utils import arg_to_list

class MultiLoader:
    @staticmethod
    def build_loaders(
        dataset: Union[str, List[str]],
        batch_size: int = 1,    # Doesn't support > 1 as probably not necessary and introduces problems like padding images to max batch size.
        check_processed: bool = True,
        data_hook: Optional[Callable] = None,
        half_precision: bool = True,
        load_data: bool = True,
        load_test_origin: bool = True,
        n_folds: Optional[int] = 5, 
        n_train: Optional[int] = None,
        n_workers: int = 1,
        p_val: float = .2,
        random_seed: int = 42,
        region: PatientRegions = 'all',
        shuffle_train: bool = True,
        test_fold: Optional[int] = None,
        transform: torchio.transforms.Transform = None,
        use_grouping: bool = False) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        datasets = arg_to_list(dataset, str)
        regions = region_to_list(region)
        if n_folds is not None and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")

        # Get dataset spacing.
        sets = {}
        prev_spacing = None
        for i, dataset in enumerate(datasets):
            sets[i] = ds.get(dataset, 'training', check_processed=check_processed) 
            spacing = sets[i].params['output-spacing']
            if prev_spacing is not None and spacing != prev_spacing:
                raise ValueError(f"Spacing must be consistent across all loader datasets. Got '{prev_spacing}' and '{spacing}'.")
            prev_spacing = spacing

        # Get all groups or samples.
        set_id_pairs = []
        for i, dataset in enumerate(datasets):
            sets[i] = ds.get(dataset, 'training', check_processed=check_processed)
            if use_grouping:
                for group_id in sets[i].list_groups(region=regions):
                    set_id_pairs.append((i, group_id))
            else:
                for sample_id in sets[i].list_samples(region=regions):
                    set_id_pairs.append((i, sample_id))

        # Shuffle groups.
        np.random.seed(random_seed)
        np.random.shuffle(set_id_pairs)

        # Split groups into folds of equal size.
        if n_folds:
            n_pairs = len(set_id_pairs)
            len_fold = int(np.floor(n_pairs / n_folds))
            folds_pairs = []
            for i in range(n_folds):
                fold_pairs = set_id_pairs[i * len_fold:(i + 1) * len_fold]
                folds_pairs.append(fold_pairs)

            # Determine train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for n_folds=5). This ensures that when we 
            # take a subset of groups (n_groups != None), we get different training groups
            # for each of the k-folds.
            train_folds = list((np.array(range(n_folds)) + (test_fold + 1)) % 5)
            train_folds.remove(test_fold)

            # Get train and test samples.
            train_samples = []
            for i in train_folds:
                for j, id in folds_pairs[i]:
                    if use_grouping:
                        sample_ids = sets[j].list_samples(group_id=id)
                        samples = [(j, id) for id in sample_ids]
                    else:
                        samples = [(j, id)]
                    train_samples += samples
            test_samples = []
            for i, id in folds_pairs[test_fold]:
                if use_grouping:
                    sample_ids = sets[i].list_samples(group_id=id)
                    samples = [(i, id) for id in sample_ids]
                else:
                    samples = [(i, id)]
                test_samples += samples
        else:
            train_samples = []
            for i, id in set_id_pairs:
                if use_grouping:
                    sample_ids = sets[i].list_samples(group_id=id)
                    samples = [(i, id) for id in sample_ids]
                else:
                    samples = [(i, id)]
                train_samples += samples

        # Take subset of train samples.
        if n_train is not None:
            if n_train > len(train_samples):
               raise ValueError(f"'n_train={n_train}' requested larger number than training samples '{len(train_samples)}'.") 
            train_samples = train_samples[:n_train]

        # Split train folds' samples into training and validation samples.
        n_train = int(len(train_samples) * (1 - p_val))
        train_train_samples = train_samples[:n_train]
        train_val_samples = train_samples[n_train:] 

        # Create train loader.
        train_ds = TrainingDataset(datasets, train_train_samples, data_hook=data_hook, half_precision=half_precision, load_data=load_data, region=regions, spacing=spacing, transform=transform)
        train_loader = DataLoader(batch_size=batch_size, dataset=train_ds, num_workers=n_workers, shuffle=shuffle_train)

        # Create validation loader.
        class_weights = np.ones(len(regions) + 1) / (len(regions) + 1)
        val_ds = TrainingDataset(datasets, train_val_samples, class_weights=class_weights, data_hook=data_hook, half_precision=half_precision, load_data=load_data, region=regions, spacing=spacing)
        val_loader = DataLoader(batch_size=batch_size, dataset=val_ds, num_workers=n_workers, shuffle=False)

        # Create test loader.
        if n_folds:
            test_ds = TestDataset(datasets, test_samples, load_origin=load_test_origin) 
            test_loader = DataLoader(batch_size=batch_size, dataset=test_ds, num_workers=n_workers, shuffle=False)
            return train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader

class TrainingDataset(Dataset):
    def __init__(
        self,
        datasets: List[str],
        samples: List[Tuple[int, int]],
        class_weights: Optional[np.ndarray] = None,
        data_hook: Optional[Callable] = None,
        half_precision: bool = True,
        load_data: bool = True,
        region: PatientRegions = 'all',
        spacing: Optional[ImageSpacing3D] = None,
        transform: torchio.transforms.Transform = None):
        self.__class_weights = class_weights
        self.__data_hook = data_hook
        self.__half_precision = half_precision
        self.__load_data = load_data
        self.__regions = region_to_list(region)
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

        if class_weights is None:
            # Calculate weights based on training data.
            region_counts = np.zeros(self.__n_channels, dtype=int)
            for ds_i, s_i in samples:
                regions = self.__sets[ds_i].sample(s_i).list_regions()
                print(ds_i, s_i)
                print(regions)
                regions = [r for r in regions if r in self.__regions]
                for region in regions:
                    region_counts[self.__region_channel_map[region]] += 1

                # If all regions are present, we can train background class.
                if len(regions) == len(self.__regions):
                    region_counts[0] += 1

            logging.info(f"Calculated region counts '{region_counts}'.")

            # Set class weights as inverse of region counts.
            assert len(np.argwhere(region_counts)) == self.__n_channels
            class_weights = 1 / region_counts

            # Normalise weight values.
            class_weights = class_weights / np.sum(class_weights)

        logging.info(f"Using class weights '{class_weights}'.")
        self.__class_weights = class_weights

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
            return desc

        # Load region data.
        sample = set.sample(s_i)
        regions = sample.list_regions()
        regions = [r for r in regions if r in self.__regions]
        input, labels = sample.pair(region=regions)

        # Apply data hook.
        if self.__data_hook is not None:
            input, labels = self.__data_hook(input, labels, spacing=self.__spacing)

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
        if len(regions) == len(self.__regions):
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
            output = self.__transform(subject)

            # Remove 'channel' dimension.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

            # Convert to numpy.
            input = input.numpy()
            label = label.numpy().astype(bool)

        # Convert dtypes.
        if self.__half_precision:
            input = input.astype(np.half)
        else:
            input = input.astype(np.single)
        label = label.astype(bool)

        # Add channel dimension - expected by pytorch.
        input = np.expand_dims(input, 0)

        return desc, input, label, mask, self.__class_weights
    
class TestDataset(Dataset):
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
