import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import List, Optional, Tuple, Union

from mymi import types
from mymi import dataset as ds
from mymi.dataset.training import TrainingDataset
from mymi.regions import RegionMap, RegionNames
from mymi.utils import arg_to_list

class MultiLoader:
    @staticmethod
    def build_loaders(
        dataset: Union[str, List[str]],
        batch_size: int = 1,
        check_processed: bool = True,
        half_precision: bool = True,
        load_data: bool = True,
        load_test_origin: bool = True,
        n_folds: Optional[int] = 5, 
        n_train: Optional[int] = None,
        n_workers: int = 1,
        random_seed: int = 42,
        shuffle_train: bool = True,
        spacing: Optional[types.ImageSpacing3D] = None,
        test_fold: Optional[int] = None,
        transform: torchio.transforms.Transform = None,
        p_val: float = .2) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        datasets = arg_to_list(dataset, str)
        if type(datasets) == str:
            datasets = [datasets]
        if n_folds is not None and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")

        # Get all samples.
        samples = []
        for i, dataset in enumerate(datasets):
            set = ds.get(dataset, 'training', check_processed=check_processed)
            for sample in set.list_samples():
                samples.append((i, sample))

        # Shuffle samples.
        np.random.seed(random_seed)
        np.random.shuffle(samples)

        # Split samples into folds of equal size.
        if n_folds:
            n_samples = len(samples)
            len_fold = int(np.floor(n_samples / n_folds))
            folds = []
            for i in range(n_folds):
                fold = samples[i * len_fold:(i + 1) * len_fold]
                folds.append(fold)

            # Determine train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for n_folds=5). This ensures that when we 
            # take a subset of samples (n_samples != None), we get different training samples
            # for each of the k-folds.
            train_folds = list((np.array(range(n_folds)) + (test_fold + 1)) % 5)
            train_folds.remove(test_fold)

            # Get train and test data.
            train_samples = []
            for i in train_folds:
                train_samples += folds[i]
            test_samples = folds[test_fold] 
        else:
            train_samples = samples

        # Take subset of train samples.
        if n_train is not None:
            if n_train > len(train_samples):
               raise ValueError(f"'n_train={n_train}' requested larger number than training samples '{len(train_samples)}'.") 
            train_samples = train_samples[:n_train]

        # Split train into NN train and validation data.
        n_nn_train = int(len(train_samples) * (1 - p_val))
        nn_train_samples = train_samples[:n_nn_train]
        nn_val_samples = train_samples[n_nn_train:] 

        # Create train loader.
        train_ds = TrainingDataset(datasets, nn_train_samples, half_precision=half_precision, load_data=load_data, spacing=spacing, transform=transform)
        train_loader = DataLoader(batch_size=batch_size, dataset=train_ds, num_workers=n_workers, shuffle=shuffle_train)

        # Create validation loader.
        val_ds = TrainingDataset(datasets, nn_val_samples, half_precision=half_precision, load_data=load_data, spacing=spacing)
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
        half_precision: bool = True,
        load_data: bool = True,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        self.__sets = [ds.get(dataset, 'training') for dataset in datasets]
        self.__half_precision = half_precision
        self.__load_data = load_data
        self.__spacing = spacing
        self.__transform = transform
        if transform:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Record number of samples.
        self.__n_samples = len(samples)

        # Map loader indices to dataset indices.
        self.__sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

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

        # Load all region data.
        sample = set.sample(s_i)
        regions = sample.list_regions()
        input, labels = sample.pair(regions=regions)

        # Create multi-class mask and label.
        # Note that using this method we may end up with multiple foreground classes for a
        # single voxel. E.g. brain/brainstem both present. Don't worry about this for now,
        # the network will just try to maximise both (and fail).
        n_channels = len(RegionNames) + 1   # Background is channel 0.
        mask = np.zeros(n_channels, dtype=bool)
        label = np.zeros((n_channels, *input.shape), dtype=bool)
        for region in regions:
            mask[RegionMap[region]] = True
            label[RegionMap[region]] = labels[region]

        # Add background information if all regions are annotated.
        if len(regions) == len(RegionNames):
            mask[0] = True
            # Collapse along channel axis - seeing if any other class is present for this voxel.
            label[0] = np.invert(label.any(axis=0))

        # Perform transform.
        if self.__transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self.__spacing[0], 0, 0, 0],
                [0, self.__spacing[1], 0, 0],
                [0, 0, self.__spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject_kwargs = { 'input': input }
            for r, d in label.items():
                subject_kwargs[r] = d
            subject = Subject({
                'input': input,
                'label': label
            })

            # Transform the subject.
            output = self.__transform(subject)

            # Extract results.
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

        return desc, input, label, mask
    
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
