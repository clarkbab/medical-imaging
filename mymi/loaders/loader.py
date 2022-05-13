import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import List, Optional, Tuple, Union

from mymi import types
from mymi import dataset as ds
from mymi.dataset.training import TrainingDataset
from mymi.geometry import get_box, get_encaps_dist_vox, get_extent_centre
from mymi.regions import get_patch_size
from mymi.transforms import point_crop_or_pad_3D

class Loader:
    @staticmethod
    def build_loaders(
        datasets: Union[str, List[str]],
        region: str,
        batch_size: int = 1,
        extract_patch: bool = False,
        half_precision: bool = True,
        load_test_origin: bool = True,
        num_folds: Optional[int] = None, 
        num_train: Optional[int] = None,
        num_workers: int = 1,
        random_seed: int = 42,
        spacing: Optional[types.ImageSpacing3D] = None,
        test_fold: Optional[int] = None,
        transform: torchio.transforms.Transform = None,
        p_val: float = .2) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        if type(datasets) == str:
            datasets = [datasets]
        if num_folds and test_fold is None:
            raise ValueError(f"'test_fold' must be specified when performing k-fold training.")
        if extract_patch and not spacing:
            raise ValueError(f"'spacing' must be specified when extracting segmentation patches.") 

        # Get all samples.
        datasets = [ds.get(d, 'training') for d in datasets]
        all_samples = []
        for ds_i, dataset in enumerate(datasets):
            samples = dataset.list_samples(regions=region)
            for s_i in samples:
                all_samples.append((ds_i, s_i))

        # Shuffle samples.
        np.random.seed(random_seed)
        np.random.shuffle(all_samples)

        # Split samples into folds.
        if num_folds:
            num_samples = len(all_samples)
            len_fold = int(np.floor(num_samples / num_folds))
            folds = []
            for i in range(num_folds):
                fold = all_samples[i * len_fold:(i + 1) * len_fold]
                folds.append(fold)

            # Determine train and test folds. Note if (e.g.) test_fold=2, then the train
            # folds should be [3, 4, 0, 1] (for num_folds=5). This ensures that when we 
            # take a subset of samples (num_samples != None), we get different training samples
            # for each of the k-folds.
            train_folds = list((np.array(range(num_folds)) + (test_fold + 1)) % 5)
            train_folds.remove(test_fold)

            # Get train and test data.
            train_samples = []
            for i in train_folds:
                train_samples += folds[i]
            test_samples = folds[test_fold] 
        else:
            train_samples = all_samples

        # Take subset of train samples.
        if num_train is not None:
            if num_train > len(train_samples):
               raise ValueError(f"'num_train={num_train}' requested larger number than training samples '{len(train_samples)}'.") 
            train_samples = train_samples[:num_train]

        # Split train into NN train and validation data.
        num_nn_train = int(len(train_samples) * (1 - p_val))
        nn_train_samples = train_samples[:num_nn_train]
        nn_val_samples = train_samples[num_nn_train:] 

        # Create train loader.
        train_ds = TrainingDataset(datasets, region, nn_train_samples, extract_patch=extract_patch, half_precision=half_precision, spacing=spacing, transform=transform)
        train_loader = DataLoader(batch_size=batch_size, dataset=train_ds, num_workers=num_workers, shuffle=True)

        # Create validation loader.
        val_ds = TrainingDataset(datasets, region, nn_val_samples, extract_patch=extract_patch, half_precision=half_precision, spacing=spacing)
        val_loader = DataLoader(batch_size=batch_size, dataset=val_ds, num_workers=num_workers, shuffle=False)

        # Create test loader.
        if num_folds:
            test_ds = TestDataset(datasets, test_samples, load_origin=load_test_origin) 
            test_loader = DataLoader(batch_size=batch_size, dataset=test_ds, num_workers=num_workers, shuffle=False)
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
        half_precision: bool = True,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        self._datasets = datasets
        self._extract_patch = extract_patch
        self._half_precision = half_precision
        self._region = region
        self._spacing = spacing
        self._transform = transform
        if transform:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Record number of samples.
        self._num_samples = len(samples)

        # Map loader indices to dataset indices.
        self._sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

    def __len__(self):
        return self._num_samples

    def __getitem__(
        self,
        index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Load data.
        ds_i, s_i = self._sample_map[index]
        dataset = self._datasets[ds_i]
        input, labels = dataset.sample(s_i).pair(regions=self._region)
        label = labels[self._region]

        # Get description.
        desc = f'{dataset.name}:{s_i}'

        # Perform transform.
        if self._transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self._spacing[0], 0, 0, 0],
                [0, self._spacing[1], 0, 0],
                [0, 0, self._spacing[2], 1],
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
            output = self._transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

            # Convert to numpy.
            input = input.numpy()
            label = label.numpy().astype(bool)

        # Extract patch.
        if self._extract_patch:
            # Augmentation may have moved all foreground voxels off the label.
            if label.sum() > 0:
                input, label = self._get_foreground_patch(input, label, desc)
            else:
                input, label = self._get_random_patch(input, label)

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert dtypes.
        if self._half_precision:
            input = input.astype(np.half)
        else:
            input = input.astype(np.single)
        label = label.astype(bool)

        return desc, input, label

    def _get_foreground_patch(
        self,
        input: np.ndarray,
        label: np.ndarray,
        desc: str) -> np.ndarray:

        # Create segmenter patch.
        centre = get_extent_centre(label)
        size = get_patch_size(self._region, self._spacing)
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
        input = point_crop_or_pad_3D(input, size, centre, fill=input.min())        
        label = point_crop_or_pad_3D(label, size, centre)

        return input, label

    def _get_random_patch(
        self,
        input: np.ndarray,
        label: np.ndarray) -> np.ndarray:
        # Choose a random voxel.
        centre = tuple(map(np.random.randint, input.shape))

        # Extract patch around centre.
        size = get_patch_size(self._region, self._spacing)
        input = point_crop_or_pad_3D(input, size, centre, fill=input.min())        
        label = point_crop_or_pad_3D(label, size, centre)

        return input, label
    
class TestDataset(Dataset):
    def __init__(
        self,
        datasets: List[TrainingDataset],
        samples: List[Tuple[int, int]],
        load_origin: bool = True):
        self._datasets = datasets
        self._load_origin = load_origin

        # Record number of samples.
        self._num_samples = len(samples)

        # Map loader indices to dataset indices.
        self._sample_map = dict(((i, sample) for i, sample in enumerate(samples)))

    def __len__(self):
        return self._num_samples

    def __getitem__(
        self,
        index: int) -> Tuple[str, str]:
        # Load data.
        ds_i, s_i = self._sample_map[index]
        set = self._datasets[ds_i]
        
        # Return 'NIFTI' location of training data.
        if self._load_origin:
            ds_name = set.sample(s_i).origin_dataset
            pat_id = set.sample(s_i).patient_id
            data = (ds_name, pat_id)
        else:
            data = (set.name, s_i) 

        return data
