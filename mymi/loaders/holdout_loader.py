import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from torchio.transforms import Transform
from tqdm import tqdm
from typing import *

from mymi import datasets as ds
from mymi.datasets.training import TrainingDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *

from .random_sampler import RandomSampler

class HoldoutLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        batch_size: int = 1,
        n_workers: int = 1,
        preload_samples: bool = True,
        random_seed: int = 42,
        regions: Optional[PatientRegions] = 'all',
        shuffle_train: bool = True,
        train_transform: Optional[Transform] = None,
        **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:

        # Create train loader.
        set = TrainingDataset(dataset)
        regions = regions_to_list(regions, literals={ 'all': set.list_regions })
        train_split = set.split('train')
        train_set = TrainingSet(train_split, regions, preload_samples=preload_samples, spacing=set.spacing, transform=train_transform) 
        if shuffle_train:
            train_sampler = RandomSampler(train_set, random_seed=random_seed)
        else:
            train_sampler = None
        train_loader = DataLoader(batch_size=batch_size, dataset=train_set, num_workers=n_workers, sampler=train_sampler, shuffle=False)

        # Create validate loader.
        val_split = set.split('validate')
        val_set = TrainingSet(val_split, regions, preload_samples=preload_samples) 
        val_loader = DataLoader(batch_size=batch_size, dataset=val_set, num_workers=n_workers, shuffle=False)

        return train_loader, val_loader, None

class TrainingSet(Dataset):
    def __init__(
        self,
        split: 'TrainingSplit',
        regions: List[PatientRegion],
        preload_samples: bool = True,
        spacing: Optional[ImageSpacing3D] = None,
        transform: torchio.transforms.Transform = None):
        self.__preload_samples = preload_samples
        self.__regions = regions
        self.__spacing = spacing
        self.__split = split
        self.__transform = transform
        if self.__transform:
            assert self.__spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Record number of samples.
        self.__sample_ids = self.__split.list_samples()
        self.__n_samples = len(self.__sample_ids)

        # Preload samples.
        if self.__preload_samples:
            logging.info(f"Preloading training samples (n={self.__n_samples}).")
            self.__inputs = []
            self.__labels = []
            self.__masks = []
            for s in tqdm(self.__sample_ids):
                sample = self.__split.sample(s)
                input = sample.input
                self.__inputs.append(input)
                label = sample.label(regions=self.__regions)
                self.__labels.append(label)
                mask = sample.mask(regions=self.__regions)
                self.__masks.append(mask)

    def __len__(self):
        return self.__n_samples

    def __getitem__(
        self,
        idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Create description:
        sample_id = self.__sample_ids[idx]
        sample = self.__split.sample(sample_id)
        desc = str(sample)

        # Load data.
        if self.__preload_samples:
            input = self.__inputs[idx]
            label = self.__labels[idx]
            mask = self.__masks[idx]
        else:
            input, label = sample.pair(regions=self.__regions)
            mask = sample.mask(regions=self.__regions)

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

        # Add channel dimension - expected by pytorch.
        input = np.expand_dims(input, 0)

        # Cast to required training types.
        input = input.astype(np.float32)

        return desc, input, label, mask
    
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
