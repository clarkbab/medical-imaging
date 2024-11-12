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

class RegLoader:
    @staticmethod
    def build_loaders(
        dataset: Union[str, List[str]],
        batch_size: int = 1,
        check_processed: bool = True,
        data_hook: Optional[Callable] = None,
        epoch: int = 0,
        include_background: bool = False,
        load_all_samples: bool = False,
        load_data: bool = True,
        load_test_origin: bool = True,
        load_train_origin: bool = False,
        n_folds: Optional[int] = None, 
        n_subfolds: Optional[int] = None,
        n_train: Optional[int] = None,
        n_workers: int = 1,
        p_val: float = .2,
        p_same: float = 0,
        preload_data: bool = True,
        random_seed: int = 0,
        regions: Optional[PatientRegions] = None,
        shuffle_samples: bool = True,
        shuffle_train: bool = True,
        test_fold: Optional[int] = None,
        test_subfold: Optional[int] = None,
        transform_train: Transform = None,
        transform_val: Transform = None,
        use_grouped_split: bool = False,
        use_grouped_test: bool = True,
        use_grouped_train: bool = False,
        use_grouped_val: bool = False,
        use_split_file: bool = False,
        **kwargs) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        datasets = arg_to_list(dataset, str)
        if n_folds is not None and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")
        logging.arg_log('Building reg-loaders', ('dataset', 'regions', 'load_all_samples', 'n_folds', 'shuffle_samples', 'test_fold', 'use_grouped_split', 'use_split_file'), (dataset, regions, load_all_samples, n_folds, shuffle_samples, test_fold, use_grouped_split, use_split_file))

        # Get dataset spacing.
        sets = []
        prev_spacing = None
        for i, dataset in enumerate(datasets):
            set = TrainingDataset(dataset, check_processed=check_processed)
            sets.append(set)
            spacing = set.params['spacing'] if 'spacing' in set.params else set.params['output-spacing']
            if prev_spacing is not None and spacing != prev_spacing:
                raise ValueError(f"Spacing must be consistent across all loader datasets. Got '{prev_spacing}' and '{spacing}'.")
            prev_spacing = spacing

        # Handle 'regions' arg.
        def all_regions():
            regions = []
            for set in sets:
                set_regions = set.list_regions()
                regions += set_regions
            regions = list(sorted(np.unique(regions)))
        regions = regions_to_list(regions, literals={ 'all': all_regions })

        # Load all samples/groups.
        samples = []
        for i, set in enumerate(sets):
            # Grouping is used when multiple 'patient-id' values belong to the same patient. E.g.
            # sample with 'group=0,patient-id=0-0' is the same patient as 'group=0,patient-id=0-1'.
            if use_grouped_split:
                # Loading all samples is required to ensure consistent train/test split per region
                # when passing different 'regions'.
                # If we pass 'Brain' for example, a different set of patients will be loaded (and different splits)
                # than if we passed 'Brainstem'. But we need a consistent test dataset of patients to evaluate (do we?).
                # Of course using a split based on all patients could leave us with no 'Brainstem' labels in the 
                # test dataset; however, in practice its usually fairly balanced.
                if load_all_samples:
                    set_samples = set.list_groups(sort_by_sample_id=True)
                else:
                    set_samples = set.list_groups(regions=regions, sort_by_sample_id=True)
            else:
                if load_all_samples:
                    set_samples = set.list_samples()
                else:
                    set_samples = set.list_samples(regions=regions)

            for sample_id in set_samples:
                samples.append((i, sample_id))

        logging.info(samples)

        # Shuffle samples.
        if shuffle_samples:
            np.random.seed(random_seed)
            np.random.shuffle(samples)

        # Split into training/testing samples.
        if n_folds is not None:     
            # Split samples into equally-sized folds for training/testing.
            assert test_fold is not None

            if use_split_file:
                raise ValueError(f"Using 'n_folds={n_folds}' randomises the train/test split, whilst 'use_split_file={use_split_file}' reads 'loader-split.csv' to determine train/test split. These methods don't work together.")

            # Split samples into folds.
            # Note that 'samples' here could actually be groups if 'use_grouped_split=True'.
            n_samples = len(samples)
            logging.info(f"Loaded {n_samples} { '(grouped) ' if use_grouped_split else '' }samples total.")
            len_fold = int(np.floor(n_samples / n_folds))
            n_samples_lost = n_samples - n_folds * len_fold
            logging.info(f"Lost {n_samples_lost} samples due to {n_folds}-fold split.")

            fold_sampleses = []
            for i in range(n_folds):
                fold_samples = samples[i * len_fold:(i + 1) * len_fold]
                logging.info(f"Putting {len(fold_samples)} samples into fold {i}.")
                fold_sampleses.append(fold_samples)

            # Determine train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for n_folds=5). This ensures that when we 
            # take a subset of groups (n_groups != None), we get different training groups
            # for each of the k-folds.
            train_folds = list((np.array(range(n_folds)) + (test_fold + 1)) % n_folds)
            train_folds.remove(test_fold)
            train_samples = list(chain(*[fold_sampleses[f] for f in train_folds]))
            logging.info(f"Found {len(train_samples)} train samples.")
            test_samples = fold_sampleses[test_fold]
            logging.info(f"Found {len(test_samples)} test samples.")

        elif use_split_file:         
            # Use 'loader-split.csv' to determine training/testing split.
            assert use_grouped_split is False

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
        else:       
            # All samples are used for training.
            train_samples = samples 
            test_samples = None

        # Split training samples into 'subfolds' - for hyper-parameter selection.
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

        # Get number of training (minus validation) samples.
        n_train_samples = int(len(train_samples) * (1 - p_val))

        # Split train samples into training/validation samples.
        val_samples = train_samples[n_train_samples:] 
        train_samples = train_samples[:n_train_samples]

        # Expand groups to samples.
        if use_grouped_split:
            # Expand training samples.
            train_samples_tmp = train_samples.copy()
            train_samples = []
            for set_i, group_id in train_samples_tmp:
                samples = sets[set_i].list_samples(group_ids=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                train_samples += samples
            logging.info(f"Expanded grouped train samples to {len(train_samples)} samples.")

            # Expand validation samples.
            val_samples_tmp = val_samples.copy()
            val_samples = []
            for set_i, group_id in val_samples_tmp:
                samples = sets[set_i].list_samples(group_ids=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                val_samples += samples
            logging.info(f"Expanded grouped val samples to {len(val_samples)} samples.")

            # Expand test samples.
            if test_samples is not None:
                test_samples_tmp = test_samples.copy()
                test_samples = []
                for set_i, group_id in test_samples_tmp:
                    samples = sets[set_i].list_samples(group_ids=group_id)
                    samples = [(set_i, sample_id) for sample_id in samples]
                    test_samples += samples
                logging.info(f"Expanded grouped test samples to {len(test_samples)} samples.")

                # Expand test sub-samples.
                if n_subfolds is not None:
                    test_subsamples_tmp = test_subsamples.copy()
                    test_subsamples = []
                    for set_i, group_id in test_subsamples_tmp:
                        samples = sets[set_i].list_samples(group_ids=group_id)
                        test_subsamples += samples
                    logging.info(f"Expanded grouped test subsamples to {len(test_subsamples)} samples.")

        # If 'use_grouped_train=True', then samples are sorted by group by default.
        # For reg, I think this will always be True, because we don't want patients
        # to span across train/val/test splits.
        if not use_grouped_train:
            np.random.shuffle(train_samples)
        if not use_grouped_val:
            np.random.shuffle(val_samples)
        if test_samples is not None and not use_grouped_test:
            np.random.shuffle(test_samples)

        logging.info(train_samples)
        logging.info(val_samples)
        if test_samples is not None:
            logging.info(test_samples)

        # Take subset of train samples.
        if n_train is not None:
            assert not use_split_file
            if n_train > len(train_samples):
               raise ValueError(f"'n_train={n_train}' requested larger number than training samples '{len(train_samples)}'.") 
            train_samples = train_samples[:n_train]

        # Filter out patients without one of the requested 'region/s'.
        if load_all_samples:
            # Filter training samples.
            train_samples_tmp = train_samples.copy()
            train_samples = []
            for set_i, sample_id in train_samples_tmp:
                sample = sets[set_i].sample(sample_id)
                if sample.has_region(regions):
                    train_samples.append((set_i, sample_id))
                    
            # Filter validation samples.
            val_samples_tmp = val_samples.copy()
            val_samples = []
            for set_i, sample_id in val_samples_tmp:
                sample = sets[set_i].sample(sample_id)
                if sample.has_region(regions):
                    val_samples.append((set_i, sample_id))

            # Filter test samples.
            test_samples_tmp = test_samples.copy() 
            test_samples = []
            for set_i, sample_id in test_samples_tmp:
                sample = sets[set_i].sample(sample_id)
                if sample.has_region(regions):
                    test_samples.append((set_i, sample_id))

        # Create train loader.
        col_fn = collate_fn if batch_size > 1 else None
        train_ds = TrainingSet(datasets, train_samples, data_hook=data_hook, include_background=include_background, load_data=load_data, load_origin=load_train_origin, preload_data=preload_data, random_seed=random_seed, regions=regions, spacing=spacing, transform=transform_train)
        if shuffle_train:
            shuffle = None
            train_sampler = RandomSampler(train_ds, epoch=epoch, random_seed=random_seed)
        else:
            shuffle = False
            train_sampler = None
        train_loader = DataLoader(batch_size=batch_size, collate_fn=col_fn, dataset=train_ds, num_workers=n_workers, sampler=train_sampler, shuffle=shuffle)

        # Create validation loader.
        if regions is not None:
            if include_background:
                # Give all classes equal weight.
                class_weights = np.ones(len(regions) + 1) / (len(regions) + 1)
            else:
                # Give all foreground classes equal weight.
                class_weights = np.ones(len(regions) + 1) / len(regions)
                class_weights[0] = 0
        else:
            class_weights = None
        val_ds = TrainingSet(datasets, val_samples, class_weights=class_weights, data_hook=data_hook, include_background=include_background, load_data=load_data, load_origin=load_train_origin, preload_data=preload_data, regions=regions, spacing=spacing, transform=transform_val)
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
        images: List[Tuple[int, int]],
        class_weights: Optional[np.ndarray] = None,
        data_hook: Optional[Callable] = None,
        include_background: bool = False,
        load_data: bool = True,
        load_origin: bool = False,
        preload_data: bool = True,
        random_seed: float = 0,
        regions: Optional[PatientRegions] = None,
        spacing: Optional[ImageSpacing3D] = None,
        transform: torchio.transforms.Transform = None) -> None:
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
        self.__sets = [TrainingDataset(d) for d in datasets]

        # Record number of images.
        self.__n_images = len(images)

        # Map loader indices to dataset indices.
        self.__image_map = dict(((i, image) for i, image in enumerate(images)))

        # Create map from region names to channels.
        if self.__regions is not None:
            self.__n_channels = len(self.__regions) + 1
            self.__region_channel_map = { 'background': 0 }
            for i, region in enumerate(self.__regions):
                self.__region_channel_map[region] = i + 1

        # Calculate region counts.
        if self.__regions is not None:
            region_counts = np.zeros(self.__n_channels, dtype=int)
            for ds_i, s_i in images:
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
                # Set class weights based on inverse frequency in the training dataset.
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
            logging.info(f"Preloading data for {self.__n_images} images.")
            self.__data = []
            for i in tqdm(range(self.__n_images)):
                # Get dataset sample.
                ds_i, s_i = self.__image_map[i]
                set = self.__sets[ds_i]

                # Load region data.
                sample = set.sample(s_i)
                if self.__regions is not None:
                    regions = sample.list_regions(only=self.__regions)
                    input, labels = sample.pair(regions=regions)
                    self.__data.append((input, labels))
                else:
                    input = sample.input
                    self.__data.append(input)

    def __len__(self):
        return self.__n_images // 2

    def __getitem__(
        self,
        index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get fixed/moving samples.
        moving_idx = 2 * index
        fixed_idx = 2 * index + 1
        moving_ds, moving_s = self.__image_map[fixed_idx]
        fixed_ds, fixed_s = self.__image_map[moving_idx]
        moving_set = self.__sets[moving_ds]
        fixed_set = self.__sets[fixed_ds]
        moving_sample = moving_set.sample(moving_s)
        fixed_sample = fixed_set.sample(fixed_s)

        # Get description.
        desc = f'{moving_set.name}:{moving_s}->{moving_set.name}:{fixed_s}'
        if not self.__load_data:
            if self.__load_origin:
                moving_origin = moving_sample.origin
                fixed_origin = fixed_sample.origin
                moving_desc = ':'.join((str(el) for el in moving_set.sample(moving_s).origin))
                fixed_desc = ':'.join((str(el) for el in fixed_set.sample(fixed_s).origin))
                desc = f'{moving_desc}->{fixed_desc}'
            return desc

        # Get sample regions.
        if self.__regions is not None:
            sample = set.sample(s_i)
            moving_regions = sample.list_regions(only=self.__regions)
            fixed_regions = sample.list_regions(only=self.__regions)

        # Load input/labels.
        if self.__preload_data:
            # Load data from cached Python list.
            if self.__regions is not None:
                moving_input, moving_labels = self.__data[moving_idx]
                fixed_input, fixed_labels = self.__data[fixed_idx]
            else:
                moving_input = self.__data[moving_idx]
                fixed_input = self.__data[fixed_idx]
        else:
            # Load data from disk.
            if self.__regions is not None:
                moving_input, moving_labels = moving_sample.pair(regions=moving_regions)
                fixed_input, fixed_labels = fixed_sample.pair(regions=fixed_regions)

        # Apply data hook.
        if self.__data_hook is not None:
            if self.__regions is not None:
                fixed_input, fixed_labels = self.__data_hook(fixed_set.name, fixed_s, fixed_input, fixed_labels, spacing=self.__spacing)
                moving_input, moving_labels = self.__data_hook(moving_set.name, moving_s, moving_input, moving_labels, spacing=self.__spacing)
            else:
                fixed_input = self.__data_hook(fixed_set.name, fixed_s, fixed_input, spacing=self.__spacing)
                moving_input = self.__data_hook(moving_set.name, moving_s, moving_input, spacing=self.__spacing)

        if self.__regions is not None:
            # Create multi-class mask and label.
            # Note that using this method we may end up with multiple foreground classes for a
            # single voxel. E.g. brain/brainstem both present. Don't worry about this for now,
            # the network will just try to maximise both (and fail).
            moving_mask = np.zeros(self.__n_channels, dtype=bool)
            moving_label = np.zeros((self.__n_channels, *moving_input.shape), dtype=bool)
            for r in moving_regions:
                moving_mask[self.__region_channel_map[r]] = True
                moving_label[self.__region_channel_map[r]] = moving_labels[r]

            fixed_mask = np.zeros(self.__n_channels, dtype=bool)
            fixed_label = np.zeros((self.__n_channels, *fixed_input.shape), dtype=bool)
            for r in fixed_regions:
                fixed_mask[self.__region_channel_map[r]] = True
                fixed_label[self.__region_channel_map[r]] = fixed_labels[r]

            # Add background class.
            # When all foreground regions are annotated, we can invert their union to return background label.
            # If a region is missing, we can't get the background label as we don't know which voxels are foreground
            # for the missing region and which are background.
            if self.__include_background and len(moving_regions) == len(self.__regions):
                moving_mask[0] = True
                moving_label[0] = np.invert(moving_label.any(axis=0))
                
            if self.__include_background and len(fixed_regions) == len(self.__regions):
                fixed_mask[0] = True
                fixed_label[0] = np.invert(fixed_label.any(axis=0))

        # Perform transform.
        if self.__transform:
            # Transform input/labels.
            affine = np.array([
                [self.__spacing[0], 0, 0, 0],
                [0, self.__spacing[1], 0, 0],
                [0, 0, self.__spacing[2], 1],
                [0, 0, 0, 1]
            ])
            moving_input = np.expand_dims(moving_input, axis=0)
            moving_input = ScalarImage(tensor=moving_input, affine=affine)
            if self.__regions is not None:
                moving_label = LabelMap(tensor=moving_label, affine=affine)
                moving_subject = Subject({
                    'input': moving_input,
                    'label': moving_label
                })
            else:
                moving_subject = Subject({
                    'input': moving_input
                })
                
            fixed_input = np.expand_dims(fixed_input, axis=0)
            fixed_input = ScalarImage(tensor=fixed_input, affine=affine)
            if self.__regions is not None:
                fixed_label = LabelMap(tensor=fixed_label, affine=affine)
                fixed_subject = Subject({
                    'input': fixed_input,
                    'label': fixed_label
                })
            else:
                fixed_subject = Subject({
                    'input': fixed_input
                })

            # Transform the subject.
            # seed = self.__random_seed + index
            # print(f"(pid={os.getpid()},index={index}) seeding transform {seed}")
            # seed_everything(seed)   # Ensure reproducibility when resuming training.
            moving_output = self.__transform(moving_subject)
            fixed_output = self.__transform(fixed_subject)

            # Remove 'channel' dimension and convert to numpy.
            moving_input = moving_output['input'].data.squeeze(0)
            moving_input = moving_input.numpy()
            fixed_input = fixed_output['input'].data.squeeze(0)
            fixed_input = fixed_input.numpy()

            if self.__regions is not None:
                moving_label = moving_output['label'].data.squeeze(0)
                moving_label = moving_label.numpy().astype(bool)
                fixed_label = fixed_output['label'].data.squeeze(0)
                fixed_label = fixed_label.numpy().astype(bool)

        # Add channel dimension - expected by pytorch.
        moving_input = np.expand_dims(moving_input, 0)
        fixed_input = np.expand_dims(fixed_input, 0)

        # Increment counter.
        self.__n_iter += 1

        if self.__regions is not None:
            return desc, moving_input, fixed_input
        else:
            return desc, moving_input, fixed_input, moving_label, fixed_label, moving_mask, fixed_mask, self.__class_weights
    
class TestSet(Dataset):
    def __init__(
        self,
        datasets: List[str],
        images: List[Tuple[int, int]],
        load_origin: bool = True):
        self.__sets = [TrainingDataset(d) for d in datasets]
        self.__load_origin = load_origin

        # Record number of samples.
        self.__n_images = len(images)

        # Map loader indices to dataset indices.
        self.__image_map = dict(((i, image) for i, image in enumerate(images)))

    def __len__(self):
        return self.__n_images // 2

    def __getitem__(
        self,
        index: int) -> Tuple[str]:
        # Get moving_fixed samples.
        moving_idx = 2 * index
        fixed_idx = 2 * index + 1
        moving_ds, moving_s = self.__image_map[fixed_idx]
        fixed_ds, fixed_s = self.__image_map[moving_idx]
        moving_set = self.__sets[moving_ds]
        fixed_set = self.__sets[fixed_ds]
        moving_sample = moving_set.sample(moving_s)
        fixed_sample = fixed_set.sample(fixed_s)

        # Get description.
        desc = f'{moving_set.name}:{moving_s}->{fixed_set.name}:{fixed_s}'
        if not self.__load_data:
            if self.__load_origin:
                moving_origin = moving_sample.origin
                fixed_origin = fixed_sample.origin
                moving_desc = ':'.join((str(el) for el in moving_set.sample(moving_s).origin))
                fixed_desc = ':'.join((str(el) for el in fixed_set.sample(fixed_s).origin))
                desc = f'{moving_desc}->{fixed_desc}'
            return desc

        return desc
