from itertools import chain
import numpy as np
from pytorch_lightning import seed_everything
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import Callable, List, Optional, Tuple, Union

from mymi.types import ImageSpacing3D, PatientRegion, PatientRegions
from mymi import dataset as ds
from mymi.dataset.training import TrainingDataset
from mymi.geometry import get_centre
from mymi import logging
from torchio.transforms import Transform
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

class MultiLoader:
    @staticmethod
    def build_loaders(
        dataset: Union[str, List[str]],
        batch_size: int = 1,
        check_processed: bool = True,
        data_hook: Optional[Callable] = None,
        epoch: int = 0,
        include_background: bool = False,
        load_data: bool = True,
        load_test_origin: bool = True,
        n_folds: Optional[int] = None, 
        n_subfolds: Optional[int] = None,
        n_train: Optional[int] = None,
        n_workers: int = 1,
        p_val: float = .2,
        random_seed: int = 0,
        region: Optional[PatientRegions] = None,
        shuffle_train: bool = True,
        test_fold: Optional[int] = None,
        test_subfold: Optional[int] = None,
        transform_train: Transform = None,
        transform_val: Transform = None,
        use_groups: bool = False,
        use_split_file: bool = False) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        datasets = arg_to_list(dataset, str)
        regions = arg_to_list(region, str)
        if n_folds is not None and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")

        # Get dataset spacing.
        sets = []
        prev_spacing = None
        for i, dataset in enumerate(datasets):
            set = TrainingDataset(dataset, check_processed=check_processed)
            sets.append(set)
            spacing = set.params['output-spacing']
            if prev_spacing is not None and spacing != prev_spacing:
                raise ValueError(f"Spacing must be consistent across all loader datasets. Got '{prev_spacing}' and '{spacing}'.")
            prev_spacing = spacing

        # Get regions if 'None'.
        if regions is None:
            regions = []
            for i, set in enumerate(sets):
                set_regions = set.list_regions()
                regions += set_regions
            regions = list(sorted(np.unique(regions)))

        # Load all samples/groups.
        # Grouping can be used when multiple 'patient-id' values in a dataset belong to the same patient,
        # e.g. replanning during treatment. It is important here that scans for the same patient don't 
        # end up in both training and testing dataset - leakage of testing information into training process!
        samples = []
        for i, set in enumerate(sets):
            if use_groups:
                for group_id in set.list_groups(region=regions):
                    samples.append((i, group_id))
            else:
                for sample_id in set.list_samples(region=regions):
                    samples.append((i, sample_id))

        # Shuffle samples.
        np.random.seed(random_seed)
        np.random.shuffle(samples)

        # Get training/testing samples.
        if n_folds is not None:     # Split samples into equally-sized folds for training/testing.
            assert test_fold is not None

            if use_split_file:
                raise ValueError(f"Using 'n_folds={n_folds}' randomises the train/test split, whilst 'use_split_file={use_split_file}' reads 'loader-split.csv' to determine train/test split. These methods don't work together.")

            # Split samples into folds.
            # Note that 'samples' here could actually be groups if 'use_groups=True'.
            n_samples = len(samples)
            len_fold = int(np.floor(n_samples / n_folds))
            n_samples_lost = n_samples - n_folds * len_fold
            logging.info(f"Lost {n_samples_lost} samples due to {n_folds}-fold split.")

            fold_sampleses = []
            for i in range(n_folds):
                fold_samples = samples[i * len_fold:(i + 1) * len_fold]
                fold_sampleses.append(fold_samples)

            # Determine train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for n_folds=5). This ensures that when we 
            # take a subset of groups (n_groups != None), we get different training groups
            # for each of the k-folds.
            train_folds = list((np.array(range(n_folds)) + (test_fold + 1)) % n_folds)
            train_folds.remove(test_fold)
            train_samples = list(chain(*[fold_sampleses[f] for f in train_folds]))
            test_samples = fold_sampleses[test_fold]

        elif use_split_file:         # Use 'loader-split.csv' to determine training/testing split.
            assert use_groups is False

            train_samples = []
            test_samples = []
            for i, set in enumerate(sets):
                # Get 'sample-id' values for this dataset.
                sample_ids = [s[1] for s in samples if s[0] == i]

                # Load patient 'loader-split.csv'.
                df = set.loader_split
                if df is None:
                    raise ValueError(f"No 'loader-split.csv' found for '{set}'. Either create this file, or set 'use_split_file=False'.")

                # Assign samples to partition based on 'loader-split.csv'.
                for _, row in df.iterrows():
                    sample_id, partition, use_origin = row['sample-id'], row['partition'], row['origin']
                    assert partition in ('train', 'test')

                    # If 'origin=True', then 'sample-id' defines ID at origin dataset.
                    if use_origin:
                        sample = set.sample(sample_id, by_origin_id=True) 
                        sample_id = sample.id

                    # Add patient to appropriate partition.
                    # Don't thrown an error if 'sample-id' in 'loader-split.csv' not found in 'sample_ids'. This could
                    # be because the number of patients used has been decreased due to 'region=...' filtering.
                    if sample_id in sample_ids:
                        sample = (i, sample_id)
                        if partition == 'train':
                            train_samples.append(sample)
                        elif partition == 'test':
                            test_samples.append(sample)

        else:       # All samples are used for testing - no validation required.
            # Note that these could be groups if 'use_groups=True'.
            train_samples = samples 

        # Split for hyper-parameter selection.
        if n_subfolds is not None:
            assert test_subfold is not None

            # Split train samples/groups into folds.
            n_samples = len(train_samples)
            len_fold = int(np.floor(n_samples / n_subfolds))
            n_samples_lost = n_samples - n_subfolds * len_fold
            logging.info(f"Lost {n_samples_lost} samples due to {n_subfolds}-subfold split.")
            fold_sampleses = []
            for i in range(n_subfolds):
                fold_samples = train_samples[i * len_fold:(i + 1) * len_fold]
                fold_sampleses.append(fold_samples)

            # Determine hyper-parameter selection train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for n_folds=5). This ensures that when we 
            # take a subset of groups (n_groups != None), we get different training groups
            # for each of the k-folds.
            train_subfolds = list((np.array(range(n_subfolds)) + (test_subfold + 1)) % n_subfolds)
            train_subfolds.remove(test_subfold)
            train_samples = list(chain(*[fold_sampleses[f] for f in train_subfolds]))
            test_subsamples = fold_sampleses[test_subfold]

        # Split 'train_samples' into training/validation samples.
        n_train_samples = int(len(train_samples) * (1 - p_val))
        if use_groups:      # Maintain grouping until after final split of data (training/validation).
            all_train_samples = train_samples
            train_samples = []
            val_samples = []

            # Expand groups to samples.
            for set_i, group_id in all_train_samples:
                samples = sets[set_i].list_samples(group_id=group_id)
                if len(train_samples) < n_train_samples:
                    train_samples += samples
                else:
                    val_samples += samples
        else:
            val_samples = train_samples[n_train_samples:] 
            train_samples = train_samples[:n_train_samples]

        # Convert test/subtest groups into samples.
        if use_groups:
            test_groups = test_samples
            test_samples = []
            for set_i, group_id in test_groups:
                samples = sets[set_i].list_samples(group_id=group_id)
                test_samples += samples

            test_subgroups = test_subsamples
            test_subsamples = []
            for set_i, group_id in test_subgroups:
                samples = sets[set_i].list_samples(group_id=group_id)
                test_subsamples += samples

        # Take subset of train samples.
        if n_train is not None:
            assert not use_split_file
            if n_train > len(train_samples):
               raise ValueError(f"'n_train={n_train}' requested larger number than training samples '{len(train_samples)}'.") 
            train_samples = train_samples[:n_train]

        # Create train loader.
        col_fn = collate_fn if batch_size > 1 else None
        train_ds = TrainingSet(datasets, regions, train_samples, data_hook=data_hook, include_background=include_background, load_data=load_data, random_seed=random_seed, spacing=spacing, transform=transform_train)
        if shuffle_train:
            shuffle = None
            train_sampler = RandomSampler(train_ds, epoch=epoch, random_seed=random_seed)
        else:
            shuffle = False
            train_sampler = None
        train_loader = DataLoader(batch_size=batch_size, collate_fn=col_fn, dataset=train_ds, num_workers=n_workers, sampler=train_sampler, shuffle=shuffle)

        # Create validation loader.
        if include_background:
            # Give all classes equal weight.
            class_weights = np.ones(len(regions) + 1) / (len(regions) + 1)
        else:
            # Give all foreground classes equal weight.
            class_weights = np.ones(len(regions) + 1) / len(regions)
            class_weights[0] = 0
        val_ds = TrainingSet(datasets, regions, val_samples, class_weights=class_weights, data_hook=data_hook, include_background=include_background, load_data=load_data, spacing=spacing, transform=transform_val)
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
        datasets: List[str],
        regions: List[PatientRegion],
        samples: List[Tuple[int, int]],
        class_weights: Optional[np.ndarray] = None,
        data_hook: Optional[Callable] = None,
        include_background: bool = False,
        load_data: bool = True,
        random_seed: float = 0,
        spacing: Optional[ImageSpacing3D] = None,
        transform: torchio.transforms.Transform = None):
        self.__class_weights = class_weights
        self.__data_hook = data_hook
        self.__include_background = include_background
        self.__load_data = load_data
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

        if class_weights is None:
            # Calculate weights based on training data.
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
