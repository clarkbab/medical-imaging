import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject

from mymi import types
from mymi.dataset.processed import ProcessedPartition

class SubsetLoader:
    @staticmethod
    def build(
        partition: ProcessedPartition,
        num_samples: int,
        batch_size: int = 1,
        half_precision: bool = True,
        num_workers: int = 1,
        regions: types.PatientRegions = 'all',
        shuffle: bool = True,
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None) -> torch.utils.data.DataLoader:
        """
        returns: a data loader that loads a subset of the samples.
        args:
            partition: the dataset partition.
            num_samples: the number of samples to load.
        kwargs:
            batch_size: the number of images in the batch.
            half_precision: load images at half precision.
            num_workers: the number of CPUs for data loading.
            regions: only load samples with (at least) one of the requested regions.
            shuffle: shuffle the data.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        ds = LoaderDataset(partition, num_samples, half_precision=half_precision, regions=regions, spacing=spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=ds, num_workers=num_workers, shuffle=shuffle)

class LoaderDataset(Dataset):
    def __init__(
        self,
        partition: ProcessedPartition,
        num_samples: int,
        half_precision: bool = True,
        regions: types.PatientRegions = 'all',
        spacing: types.ImageSpacing3D = None,
        transform: torchio.transforms.Transform = None):
        """
        args:
            partition: the dataset partition.
            num_samples: the number of samples to load.
        kwargs:
            half_precision: load images at half precision.
            regions: only load samples with (at least) one of the requested regions.
            spacing: the voxel spacing.
            transform: transformations to apply.
        """
        self._half_precision = half_precision
        self._num_samples = num_samples
        self._partition = partition
        self._regions = regions
        self._spacing = spacing
        self._transform = transform
        if transform:
            assert spacing is not None, 'Spacing is required when transform applied to dataloader.'

        # Filter samples by requested regions.
        samples = partition.list_samples(regions=regions)

        # Record number of samples.
        if num_samples > len(samples):
            raise ValueError(f"Subset loader requested '{num_samples}' samples with regions '{regions}', only '{len(samples)}' found.")

        # Map loader indices to dataset indices.
        self._index_map = dict(zip(range(self._num_samples), samples))

    def __len__(self):
        """
        returns: number of samples in the partition.
        """
        return self._num_samples

    def __getitem__(
        self,
        index: int):
        """
        returns: an (input, label) pair from the dataset.
        args:
            index: the item to return.
        """
        # Load data.
        input, label = self._partition.sample(self._index_map[index]).pair(regions=self._regions)

        # Perform transform.
        if self._transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = dict((r, np.expand_dims(d, axis=0)) for r, d in label.items())

            # Create 'subject'.
            affine = np.array([
                [self._spacing[0], 0, 0, 0],
                [0, self._spacing[1], 0, 0],
                [0, 0, self._spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = dict((r, LabelMap(tensor=d, affine=affine)) for r, d in label.items())
            subject_kwargs = { 'input': input }
            for r, d in label.items():
                subject_kwargs[r] = d
            subject = Subject(**subject_kwargs)

            # Transform the subject.
            output = self._transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = dict((r, output[r].data.squeeze(0)) for r in label.keys()) 

            # Convert to numpy.
            input = input.numpy()
            label = dict((r, d.numpy()) for r, d in label.items())

        # Add 'channel' dimension.
        input = np.expand_dims(input, axis=0)

        # Convert dtypes.
        if self._half_precision:
            input = input.astype(np.half)
        else:
            input = input.astype(np.single)
        label = dict((r, d.astype(np.bool)) for r, d in label.items())

        return input, label
