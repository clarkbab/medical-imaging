import numpy as np
import os
import pandas as pd
from scipy.ndimage import binary_dilation
from time import sleep
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import List, Optional, Tuple, Union

from mymi import config
from mymi import datasets as ds
from mymi.datasets.nifti import NiftiDataset
from mymi.datasets.training import TrainingDataset
from mymi.geometry import get_box, centre_of_extent
from mymi import logging
from mymi.metrics import get_encaps_dist_vox
from mymi.regions import get_region_patch_size
from mymi.transforms import point_crop_or_pad, resample, top_crop_or_pad
from mymi import typing
from mymi.utils import append_row

class Loader:
    @staticmethod
    def build_loaders(
        datasets: Union[str, List[str]],
        region: str,
        batch_size: int = 1,
        check_processed: bool = True,
        extract_patch: bool = False,
        load_data: bool = True,
        load_test_origin: bool = True,
        n_folds: Optional[int] = 5, 
        n_train: Optional[int] = None,
        n_workers: int = 1,
        random_seed: int = 0,
        shuffle_train: bool = True,
        spacing: Optional[typing.ImageSpacing3D] = None,
        test_fold: Optional[int] = None,
        transform: torchio.transforms.Transform = None,
        use_seg_run: bool = False,
        p_val: float = .2) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        if type(datasets) == str:
            datasets = [datasets]
        if n_folds and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")
        if extract_patch and not spacing:
            raise ValueError(f"'spacing' must be specified when extracting segmentation patches.") 

        if use_seg_run:
            ds_map = dict((d.replace('LOC', 'SEG'), i) for i, d in enumerate(datasets))
            datasets = [ds.get(d, 'training', check_processed=check_processed) for d in datasets]

            # Load seg run split.
            model = (f'segmenter-{region}-v2', f'clinical-fold-{test_fold}-samples-{n_train}')
            runpath = os.path.join(config.directories.runs, *model)
            runs = os.listdir(runpath)
            if len(runs) == 0:
                raise ValueError("Big problem, no run data for '{model}'.")
            run = runs[-1]
            manpath = os.path.join(runpath, run, 'loader-manifest.csv')
            df = pd.read_csv(manpath)
            df['dataset'] = df['dataset'].replace(ds_map)

            # Create train loader.
            nn_train_samples = list(df[df['loader'] == 'train'][['dataset', 'sample-id']].itertuples(index=False, name=None))
            print(nn_train_samples)
            print(f'before train: {len(nn_train_samples)}')
            before = len(nn_train_samples)
            nn_train_samples = list(filter(lambda p: not (p[0] == 1 and p[1] == 87), nn_train_samples))
            print(f'after train: {len(nn_train_samples)}')
            after = len(nn_train_samples)
            assert after == before or after == before - 1
            train_ds = TrainingDataset(datasets, region, nn_train_samples, extract_patch=extract_patch, load_data=load_data, spacing=spacing, transform=transform)
            train_loader = DataLoader(batch_size=batch_size, dataset=train_ds, num_workers=n_workers, shuffle=shuffle_train)

            # Create validation loader.
            nn_val_samples = list(df[df['loader'] == 'validate'][['dataset', 'sample-id']].itertuples(index=False, name=None))
            print(nn_val_samples)
            print(f'before val: {len(nn_val_samples)}')
            before = len(nn_val_samples)
            nn_val_samples = list(filter(lambda p: not (p[0] == 1 and p[1] == 87), nn_val_samples))
            print(f'after val: {len(nn_val_samples)}')
            after = len(nn_val_samples)
            assert after == before or after == before - 1
            val_ds = TrainingDataset(datasets, region, nn_val_samples, extract_patch=extract_patch, load_data=load_data, spacing=spacing)
            val_loader = DataLoader(batch_size=batch_size, dataset=val_ds, num_workers=n_workers, shuffle=False)

            # Create test loader.
            test_samples = list(df[df['loader'] == 'test'][['dataset', 'sample-id']].itertuples(index=False, name=None))
            print(test_samples)
            print(f'before test: {len(test_samples)}')
            before = len(test_samples)
            test_samples = list(filter(lambda p: not (p[0] == 1 and p[1] == 87), test_samples))
            print(f'after test: {len(test_samples)}')
            after = len(test_samples)
            assert after == before or after == before - 1
            test_ds = TestDataset(datasets, test_samples, load_origin=load_test_origin) 
            test_loader = DataLoader(batch_size=batch_size, dataset=test_ds, num_workers=n_workers, shuffle=False)

            return train_loader, val_loader, test_loader

        # Get all samples.
        datasets = [ds.get(d, 'training', check_processed=check_processed) for d in datasets]
        all_samples = []
        for ds_i, dataset in enumerate(datasets):
            samples = dataset.list_samples(region=region)
            for s_i in samples:
                all_samples.append((ds_i, s_i))

        # Shuffle samples.
        np.random.seed(random_seed)
        np.random.shuffle(all_samples)

        # Split samples into folds of equal size.
        if n_folds:
            n_samples = len(all_samples)
            len_fold = int(np.floor(n_samples / n_folds))
            folds = []
            for i in range(n_folds):
                fold = all_samples[i * len_fold:(i + 1) * len_fold]
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
            train_samples = all_samples

        # Take subset of train samples.
        if n_train is not None:
            logging.info(n_train)
            if n_train > len(train_samples):
               raise ValueError(f"'n_train={n_train}' requested larger number than training samples '{len(train_samples)}'.") 
            train_samples = train_samples[:n_train]

        # Split train into NN train and validation data.
        n_nn_train = int(len(train_samples) * (1 - p_val))
        nn_train_samples = train_samples[:n_nn_train]
        nn_val_samples = train_samples[n_nn_train:] 

        # Create train loader.
        train_ds = TrainingDataset(datasets, region, nn_train_samples, extract_patch=extract_patch, load_data=load_data, spacing=spacing, transform=transform)
        train_loader = DataLoader(batch_size=batch_size, dataset=train_ds, num_workers=n_workers, shuffle=shuffle_train)

        # Create validation loader.
        val_ds = TrainingDataset(datasets, region, nn_val_samples, extract_patch=extract_patch, load_data=load_data, spacing=spacing)
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
        datasets: List[TrainingDataset],
        region: str,
        samples: List[Tuple[int, int]],
        extract_patch: bool = False,
        load_data: bool = True,
        spacing: typing.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        self.__datasets = datasets
        self.__extract_patch = extract_patch
        self.__load_data = load_data
        self.__region = region
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
        dataset = self.__datasets[ds_i]

        # Get description.
        desc = f'{dataset.name}:{s_i}'
        if not self.__load_data:
            return desc

        # Load data.
        try:
            input, labels = dataset.sample(s_i).pair(region=self.__region)
            label = labels[self.__region]
        except EOFError:
            # Data is being copied by another process.
            sleep(60)
            input, labels = dataset.sample(s_i).pair(region=self.__region)
            label = labels[self.__region]
        except ValueError as e:
            # Pull in data on the fly.
            if dataset.name in ('PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC'):
                logging.error(f"Missing region '{self.__region}' for sample '{dataset.sample(s_i)}'. Pulling in from NIFTI...")

                # Load data.
                orig_dataset, orig_pat_id = dataset.sample(s_i).origin
                orig_pat = NiftiDataset(orig_dataset).patient(orig_pat_id)
                orig_spacing = orig_pat.ct_spacing
                input = orig_pat.ct_data 
                label = orig_pat.region_data(region=self.__region)[self.__region]
                
                # Process data.
                output_size = (128, 128, 150)
                output_spacing = (4, 4, 4)
                dilate_regions = ('BrachialPlexus_L', 'BrachialPlexus_R', 'Cochlea_L', 'Cochlea_R', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R')
                dilate_iter = 3
                input = resample(input, spacing=orig_spacing, output_spacing=output_spacing)
                label = resample(label, spacing=orig_spacing, output_spacing=output_spacing)
                input = top_crop_or_pad(input, output_size)
                label = top_crop_or_pad(label, output_size)
                if self.__region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)
                
                # Save input/label to dataset.
                filepath = os.path.join(dataset.path, 'data', 'inputs', f'{s_i}.npz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.savez_compressed(filepath, data=input)
                logging.info(f"Saved new input to filepath:{filepath}.")

                filepath = os.path.join(dataset.path, 'data', 'labels', self.__region, f'{s_i}.npz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.savez_compressed(filepath, data=label)
                logging.info(f"Saved new label to filepath:{filepath}.")
            else:
                raise e

        if input.shape != label.shape:
            raise ValueError(f"Should delete label {ds_i}:{s_i}.")

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

        # Extract patch.
        if self.__extract_patch:
            # Augmentation may have moved all foreground voxels off the label.
            if label.sum() > 0:
                input, label = self.__get_foreground_patch(input, label)
            else:
                input, label = self.__get_random_patch(input, label)

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert dtypes.
        input = input.astype(np.single)
        label = label.astype(bool)

        return desc, input, label

    def __get_foreground_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:

        # Create segmenter patch.
        centre = centre_of_extent(label)
        size = get_region_patch_size(self.__region, self.__spacing)
        min, max = get_box(centre, size)

        # Squash to label size.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.array(max)
        for i in range(len(max)):
            if max[i] > label.shape[i] - 1:
                max[i] = label.shape[i] - 1

        # Create label from patch.
        label_patch = np.zeros_like(label, dtype=bool)
        slices = tuple([slice(l, h + 1) for l, h in zip(min, max)])
        label_patch[slices] = True

        # Get encapsulation distance between patch and label.
        dist = get_encaps_dist_vox(label_patch, label)
        if np.any(np.array(dist) > 0):
            pass
            # raise ValueError(f"Segmentation patch doesn't encapsulate label for sample '{desc}', region '{self._region}'.")

        # Translate patch centre whilst maintaining encapsulation.
        t = tuple((np.random.randint(-d, d + 1) for d in np.abs(dist)))
        centre = tuple(np.array(centre) + t)

        # Extract segmentation patch.
        input = point_crop_or_pad(input, size, centre, fill=input.min())        
        label = point_crop_or_pad(label, size, centre)

        return input, label

    def __get_random_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:
        # Choose a random voxel.
        centre = tuple(map(np.random.randint, input.shape))

        # Extract patch around centre.
        size = get_region_patch_size(self.__region, self.__spacing)
        input = point_crop_or_pad(input, size, centre, fill=input.min())        
        label = point_crop_or_pad(label, size, centre)

        return input, label
    
class TestDataset(Dataset):
    def __init__(
        self,
        datasets: List[TrainingDataset],
        samples: List[Tuple[int, int]],
        load_origin: bool = True):
        self.__datasets = datasets
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
        set = self.__datasets[ds_i]
        
        if self.__load_origin:
            # Return 'NIFTI' location of training sample.
            desc = ':'.join((str(el) for el in set.sample(s_i).origin))
        else:
            desc = f'{set.name}:{s_i}'

        return desc
