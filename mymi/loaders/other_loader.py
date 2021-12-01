import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import List, Tuple, Union

from mymi import types
from mymi.dataset.training import TrainingPartition

class OtherLoader:
    @staticmethod
    def build(
        partitions: Union[TrainingPartition, List[TrainingPartition]],
        batch_size: int = 1,
        num_workers: int = 1,
        precision: str = 'single',
        shuffle: bool = True,
        transform: torchio.transforms.Transform = None) -> torch.utils.data.DataLoader:
        """
        returns: a data loader.
        args:
            partitions: the dataset partitions, e.g. 'train' partitions from both 'HN1' and 'HNSCC' datasets.
        kwargs:
            batch_size: the number of images in the batch.
            half_precision: load images at half precision.
            num_workers: the number of CPUs for data loading.
            regions: only load samples with (at least) one of the requested regions.
            shuffle: shuffle the data.
            transform: the transform to apply.
        """
        if type(partitions) == TrainingPartition:
            partitions = [partitions]

        # Create dataset object.
        ds = LoaderDataset(partitions, precision=precision, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=ds, num_workers=num_workers, shuffle=shuffle)

class LoaderDataset(Dataset):
    def __init__(
        self,
        partitions: List[TrainingPartition],
        precision: bool = True,
        transform: torchio.transforms.Transform = None):
        """
        args:
            partitions: the dataset partitions.
        kwargs:
            transform: transformations to apply.
        """
        self._precision = precision
        self._partitions = partitions
        self._transform = transform

        index = 0
        map_tuples = []
        for i, partition in enumerate(partitions):
            # Filter samples by requested regions.
            samples = partition.list_samples()
            for sample in samples:
                map_tuples.append((index, (i, sample)))
                index += 1

        # Record number of samples.
        self._num_samples = index

        # Map loader indices to dataset indices.
        self._index_map = dict(map_tuples)

    def __len__(self):
        """
        returns: number of samples in the partition.
        """
        return self._num_samples

    def __getitem__(
        self,
        index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns: an (input, label) pair from the dataset.
        args:
            index: the item to return.
        """
        # Load data.
        p_idx, s_idx = self._index_map[index]
        part = self._partitions[p_idx]
        input = part.sample(s_idx).input()

        # Get description.
        desc = f'{part.dataset.name}:{part.name}:{s_idx}'

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert dtypes.
        if self._precision == 'bool':
            input = input.astype(bool)
        elif self._precision == 'half':
            input = input.astype(np.half)
        elif self._precision == 'single':
            input = input.astype(np.single)

        return desc, input
