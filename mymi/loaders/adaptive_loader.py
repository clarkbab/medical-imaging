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

from mymi.typing import ImageSpacing3D, PatientRegion, PatientRegions
from mymi.datasets.training_adaptive import TrainingAdaptiveDataset
from mymi.geometry import get_centre
from mymi import logging
from mymi.regions import regions_to_list
from torchio.transforms import Transform
from mymi.transforms import centre_crop_or_pad
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
        for c in range(len(input)):     # Perform pad separately for each channel as 'centre_crop_or_pad' hasn't been written.
            input_c = centre_crop_or_pad(input[c], max_size)
            input_cs.append(input_c)
        input = np.stack(input_cs, axis=0)
        inputs.append(input)
        label_cs = []
        for c in range(len(label)): 
            label_c = centre_crop_or_pad(label[c], max_size)
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

class AdaptiveLoader:
    @staticmethod
    def build_loaders(
        dataset: Union[str, List[str]],
        batch_size: int = 1,
        check_processed: bool = True,
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
        use_split_file: bool = False,
        **kwargs) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        logging.arg_log('Building adaptive loaders', ('dataset', 'region', 'load_all_samples', 'n_folds', 'shuffle_samples', 'test_fold', 'use_grouping', 'use_split_file'), (dataset, region, load_all_samples, n_folds, shuffle_samples, test_fold, use_grouping, use_split_file))
        datasets = arg_to_list(dataset, str)
        regions = regions_to_list(region)
        if n_folds is not None and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")

        # Get dataset spacing.
        sets = []
        prev_spacing = None
        for i, dataset in enumerate(datasets):
            set = TrainingAdaptiveDataset(dataset, check_processed=check_processed)
            sets.append(set)
            spacing = set.params['spacing']
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
            n_samples = len(samples)
            logging.info(f"Loaded {n_samples} samples total.")
            len_fold = int(np.floor(n_samples / n_folds))
            n_samples_lost = n_samples - n_folds * len_fold
            logging.info(f"Lost {n_samples_lost} samples ({samples[-n_samples_lost:]}) due to {n_folds}-fold split.")

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

            # Expand validation samples.
            val_samples_tmp = val_samples.copy()
            val_samples = []
            for set_i, group_id in val_samples_tmp:
                samples = sets[set_i].list_samples(group_id=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                val_samples += samples

            # Expand test samples.
            test_samples_tmp = test_samples.copy()
            test_samples = []
            for set_i, group_id in test_samples_tmp:
                samples = sets[set_i].list_samples(group_id=group_id)
                samples = [(set_i, sample_id) for sample_id in samples]
                test_samples += samples

            # Expand test sub-samples.
            if n_subfolds is not None:
                test_subsamples_tmp = test_subsamples.copy()
                test_subsamples = []
                for set_i, group_id in test_subsamples_tmp:
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
        train_ds = TrainingSet(datasets, train_samples, include_background=include_background, load_data=load_data, preload_data=preload_data, random_seed=random_seed, spacing=spacing, transform=transform_train, use_frequency_weighting=True)
        if shuffle_train:
            shuffle = None
            train_sampler = RandomSampler(train_ds, epoch=epoch, random_seed=random_seed)
        else:
            shuffle = False
            train_sampler = None
        train_loader = DataLoader(batch_size=batch_size, collate_fn=col_fn, dataset=train_ds, num_workers=n_workers, sampler=train_sampler, shuffle=shuffle)

        # Create validation loader.
        val_ds = TrainingSet(datasets, val_samples, include_background=include_background, load_data=load_data, preload_data=preload_data, spacing=spacing, transform=transform_val)
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
        samples: List[Tuple[int, int]],
        include_background: bool = False,
        load_data: bool = True,
        preload_data: bool = False,
        random_seed: float = 0,
        spacing: Optional[ImageSpacing3D] = None,
        transform: torchio.transforms.Transform = None,
        use_frequency_weighting: bool = True):
        if transform is not None:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'
        self.__load_data = load_data
        self.__preload_data = preload_data
        self.__random_seed = random_seed
        self.__spacing = spacing
        self.__transform = transform
        
        # Load datasets.
        self.__sets = [TrainingAdaptiveDataset(d) for d in datasets]

        # Record number of samples.
        self.__n_samples = len(samples)

        # Map loader indices to dataset indices.
        self.__sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

        # Assumes a single dataset.
        regions = self.__sets[0].list_regions()

        if use_frequency_weighting:
            # Get region counts.
            counts = np.zeros(len(regions), dtype=np.float32)
            for ds_i, s_i in samples:
                sample_regions = self.__sets[ds_i].sample(s_i).list_regions(regions=regions)
                samples_counts = np.array([1 if r in sample_regions else 0 for r in regions], dtype=np.float32)
                counts += samples_counts
            logging.info(f"Region counts: {counts}.")

            # Calculate frequencies.
            inv_counts = counts.copy()
            for i in range(len(inv_counts)):
                if inv_counts[i] == 0:
                    inv_counts[i] = 0
                else:
                    inv_counts[i] = 1 / inv_counts[i]
            logging.info(f"Inverse region counts: {inv_counts}.")

            # Normalise inverse counts.
            self.__class_weights = inv_counts / inv_counts.sum()
        else:
            # Load first label.
            ds_i, s_i = samples[0]
            label = self.__sets[ds_i].sample(s_i).label 
            weights = np.ones(label.shape[0], dtype=np.float32)
            self.__class_weights = weights / weights.sum()

        # Add background weight.
        self.__class_weights = np.insert(self.__class_weights, 0, 0)

        logging.info(f"Using weights '{self.__class_weights}'.")

        # Preload data.
        if preload_data:
            logging.info(f"Preloading data for {self.__n_samples} samples.")
            self.__data = []
            for i in tqdm(range(self.__n_samples)):
                # Get dataset sample.
                ds_i, s_i = self.__sample_map[i]
                set = self.__sets[ds_i]

                # Load region data.
                sample = set.sample(s_i)
                input, label = sample.pair
                self.__data.append((input, label))

    def __len__(self):
        return self.__n_samples

    def __getitem__(
        self,
        index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get description.
        ds_i, s_i = self.__sample_map[index]
        set = self.__sets[ds_i]
        desc = f'{set.name}:{s_i}'
        if not self.__load_data:
            return desc

        # Load input/label data.
        sample = set.sample(s_i)
        if self.__preload_data:
            input, label = self.__data[index]
        else:
            input, label = sample.pair

        # Create mask from label.
        mask = label.sum(axis=(1, 2, 3)).astype(np.bool_)

        # Perform transform.
        if self.__transform:
            # Transform input/labels.
            affine = np.array([
                [self.__spacing[0], 0, 0, 0],
                [0, self.__spacing[1], 0, 0],
                [0, 0, self.__spacing[2], 1],
                [0, 0, 0, 1]
            ])
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
            input = output['input'].data
            label = output['label'].data

            # Convert to numpy.
            input = input.numpy()
            label = label.numpy().astype(bool)

        return desc, input, label, mask, self.__class_weights
    
class TestSet(Dataset):
    def __init__(
        self,
        datasets: List[str],
        samples: List[Tuple[int, int]],
        load_origin: bool = True):
        self.__sets = [TrainingAdaptiveDataset(d) for d in datasets]
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
