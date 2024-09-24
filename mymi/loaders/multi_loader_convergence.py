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

from mymi.types import Spacing3D, PatientRegion, PatientRegions
from mymi import dataset as ds
from mymi.dataset.training import TrainingDataset
from mymi.geometry import get_centre
from mymi import logging
from torchio.transforms import Transform
from mymi.regions import region_to_list
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

class MultiLoaderConvergence:
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
        n_folds: Optional[int] = None, 
        n_subfolds: Optional[int] = None,
        n_train: Optional[int] = None,
        n_workers: int = 1,
        p_val: float = .2,
        preload_data: bool = False,
        random_seed: int = 0,
        region: Optional[PatientRegions] = None,
        shuffle_samples: bool = True,
        shuffle_train: bool = True,
        test_fold: Optional[int] = None,
        test_subfold: Optional[int] = None,
        transform_train: Transform = None,
        transform_val: Transform = None,
        use_grouping: bool = False,
        use_pooling_hack: bool = False,
        use_pooling_hack_single: bool = False,
        use_split_file: bool = False,
        **kwargs) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        datasets = arg_to_list(dataset, str)
        regions = region_to_list(region)
        if n_folds is not None and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")
        logging.arg_log('Building multi-loaders', ('dataset', 'regions', 'load_all_samples', 'n_folds', 'shuffle_samples', 'test_fold', 'use_grouping', 'use_split_file'), (dataset, regions, load_all_samples, n_folds, shuffle_samples, test_fold, use_grouping, use_split_file))

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

        # Get regions if 'None'.
        if regions is None:
            regions = []
            for set in sets:
                set_regions = set.list_regions()
                regions += set_regions
            regions = list(sorted(np.unique(regions)))

        # Load all samples/groups.
        samples = []
        for i, set in enumerate(sets):
            # Grouping is used when multiple 'patient-id' values belong to the same patient. E.g.
            # sample with 'group=0,patient-id=0-0' is the same patient as 'group=0,patient-id=0-1'.
            if use_grouping:
                # Loading all samples is required to ensure consistent train/test split per region
                # when passing different 'regions'.
                if load_all_samples:
                    set_samples = set.list_groups(sort_by_sample_id=True)
                else:
                    set_samples = set.list_groups(region=regions, sort_by_sample_id=True)
            else:
                if load_all_samples:
                    set_samples = set.list_samples()
                else:
                    set_samples = set.list_samples(region=regions)

            for sample_id in set_samples:
                samples.append((i, sample_id))

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
            # Note that 'samples' here could actually be groups if 'use_grouping=True'.
            n_samples = len(samples)
            logging.info(f"Loaded {n_samples} { '(grouped) ' if use_grouping else '' }samples total.")
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
            assert use_grouping is False

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
        if use_grouping:
            # Expand training samples.
            train_samples_tmp = train_samples.copy()
            train_samples = []
            for set_i, group_id in train_samples_tmp:
                samples = sets[set_i].list_samples(group_id=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                train_samples += samples
            logging.info(f"Expanded grouped train samples to {len(train_samples)} samples.")

            # Expand validation samples.
            val_samples_tmp = val_samples.copy()
            val_samples = []
            for set_i, group_id in val_samples_tmp:
                samples = sets[set_i].list_samples(group_id=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                val_samples += samples
            logging.info(f"Expanded grouped val samples to {len(val_samples)} samples.")

            # Expand test samples.
            test_samples_tmp = test_samples.copy()
            test_samples = []
            for set_i, group_id in test_samples_tmp:
                samples = sets[set_i].list_samples(group_id=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                test_samples += samples
            logging.info(f"Expanded grouped test samples to {len(test_samples)} samples.")

            # Expand test sub-samples.
            if n_subfolds is not None:
                test_subsamples_tmp = test_subsamples.copy()
                test_subsamples = []
                for set_i, group_id in test_subsamples_tmp:
                    samples = sets[set_i].list_samples(group_id=group_id)
                    test_subsamples += samples
                logging.info(f"Expanded grouped test subsamples to {len(test_subsamples)} samples.")

        if use_pooling_hack:
            n_train_before = len(train_samples)
            n_val_before = len(val_samples)
            n_test_before = len(test_samples)

            # Get pre-treatment samples from 'test' dataset.
            pt_test_samples = [s for i, s in enumerate(test_samples) if i % 2 == 0]

            # Assert that all of these samples point to a pre-treatment scan.
            for d, s in pt_test_samples:
                sample = sets[d].sample(s)
                pat_id = sample.origin[1]
                if '-0' not in pat_id:
                    raise ValueError(f"Pooling hack - extracted test sample {sample} was not pre-treatment scan.")

            # How to add pre-treatment test scans to training dataset?
            # 1. We shouldn't introduce pre-treatment test scans into the validation dataset as the model never "sees"
            # these during training.
            # 2. When we replace 22 (pre/mid-treatment) samples with 22 pre-treatment test samples, we introduce
            # 22 new patients into the training dataset, and only remove 11 patients from the training dataset
            # as we're removing both pre and mid-treatment scans - to try to balance patient numbers as best as possible.
            # If we only removed pre-treatment scans from the training dataset, no patients would be removed as
            # their mid-treatment scans would remain. So the adaptive-train model has an extra 11 patients in the 
            # training dataset in comparison to the adaptive-input model, however it has a smaller number of mid-treatment
            # scans. I would assume the difference between CT scans of separate patients to be larger than the difference
            # between pre/mid-treatment scans of the same patients, so perhaps we've minimised the differences in 
            # training datasets as best we can.
            # 3. Another method would be to train a separate model test patient, just adding their pre-treatment test
            # scan to the data pool. But this creates a looot of models.

            # More thinking...
            # We should take 22 samples (22 patients) from the pre-treatment test dataset and add them to the training
            # dataset. However, we should only remove 21 samples (11 patients) from the training dataset. This means
            # that the "pooling" model sees 1 more sample during training (176 samples), whilst the "adaptive" model
            # sees 1 more sample during inference.

            # Replace train samples with pre-treatment test samples.
            n_test_samples = len(pt_test_samples)
            n_remove = n_test_samples
            if use_pooling_hack_single:
                n_remove -= 1
            logging.info(f"Pooling hack - replacing {n_remove} training samples with {n_test_samples} pre-treatment test samples.")
            train_samples[:n_remove] = pt_test_samples
            # for i, s in enumerate(pt_test_samples):
            #     # Replace every second training sample - so that we're replacing pre-treatment scans only.
            #     pt_i = i * 2
            #     pt_d, pt_s = train_samples[pt_i]
            #     sample = sets[pt_d].sample(pt_s)
            #     pat_id = sample.origin[1]
            #     if '-0' not in pat_id:
            #         raise ValueError(f"Pooling hack - replaced training sample {sample} was not pre-treatment scan.")
            #     train_samples[pt_i] = s

            # # Add pre-treatment test samples to train/val set.
            # logging.info(f"Pooling hack - adding {n_samples_to_train} pre-treatment test samples to train samples.")
            # train_samples += pt_test_samples_to_train
            # logging.info(f"Pooling hack - adding {n_samples_to_val} pre-treatment test samples to val samples.")
            # val_samples += pt_test_samples_to_val

            # # Add pre-treatment test samples to train set.
            # logging.info(f"Pooling hack - adding {n_samples} pre-treatment test samples to train samples.")
            # train_samples += pt_test_samples

            # Check that train/val/test dataset sizes haven't changed.
            # We should not take notice of pre-treatment test evaluation scores as they were included in training -
            # but we never did anyway.
            if use_pooling_hack_single:
                assert len(train_samples) == n_train_before + 1
            else:
                assert len(train_samples) == n_train_before
            assert len(val_samples) == n_val_before
            assert len(test_samples) == n_test_before

            logging.info(f"Pooling hack - train samples: {len(train_samples)}")
            logging.info(f"Pooling hack - val samples: {len(val_samples)}")
            logging.info(f"Pooling hack - test samples: {len(test_samples)}")

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
        train_ds = TrainingSet(datasets, regions, train_samples, data_hook=data_hook, include_background=include_background, load_data=load_data, preload_data=preload_data, random_seed=random_seed, spacing=spacing, transform=transform_train)
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
        val_ds = TrainingSet(datasets, regions, val_samples, class_weights=class_weights, data_hook=data_hook, include_background=include_background, load_data=load_data, preload_data=preload_data, spacing=spacing, transform=transform_val)
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
        preload_data: bool = False,
        random_seed: float = 0,
        spacing: Optional[Spacing3D] = None,
        transform: torchio.transforms.Transform = None):
        self.__class_weights = class_weights
        self.__data_hook = data_hook
        self.__include_background = include_background
        self.__load_data = load_data
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

            # # Make smallest weight equal to 1.
            # min_weight = np.min(class_weights[1:])
            # class_weights = class_weights / min_weight

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
                regions = sample.list_regions(only=self.__regions)
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
            return desc

        # Get sample regions.
        sample = set.sample(s_i)
        regions = sample.list_regions(only=self.__regions)

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
