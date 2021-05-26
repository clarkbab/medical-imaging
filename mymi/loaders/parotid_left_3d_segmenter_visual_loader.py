import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from torchio import LabelMap, ScalarImage, Subject
from typing import *

from mymi import config

class ParotidLeft3DSegmenterVisualLoader:
    @staticmethod
    def build(num_batches=5, seed=42, batch_size=1, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        returns: a data loader.
        kwargs:
            batch_size: the number of images in the batch.
            num_batches: how many batches this loader should generate.
            seed: random number generator seed.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DSegmenterVisualDataset(raw_input=raw_input, raw_label=raw_label, spacing=spacing, transform=transform)

        # Create sampler.
        sampler = ParotidLeft3DSegmenterVisualSampler(dataset, num_batches * batch_size, seed)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)

class ParotidLeft3DSegmenterVisualDataset(Dataset):
    def __init__(self, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        kwargs:
            raw_input: return the raw input data loaded from disk, in addition to transformed data.
            raw_label: return the raw label data loaded from disk, in addition to transformed data.
            spacing: the voxel spacing of the data on disk.
            transform: transformations to apply.
        """
        self.raw_input = raw_input
        self.raw_label = raw_label
        self.spacing = spacing
        self.transform = transform
        if transform:
            assert spacing, 'Spacing is required when transform applied to dataloader.'

        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        folder_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'validate')
        self.samples = np.reshape([os.path.join(folder_path, p) for p in sorted(os.listdir(folder_path))], (-1, 2))
        self.num_samples = len(self.samples)

    def __len__(self):
        """
        returns: number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        returns: an (input, label) pair from the dataset.
        idx: the item to return.
        """
        # Get data and label paths.
        input_path, label_path = self.samples[idx]

        # Load data and label.
        f = open(input_path, 'rb')
        input = np.load(f)
        f = open(label_path, 'rb')
        label = np.load(f)

        # Perform transform.
        if self.transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1], 0, 0],
                [0, 0, self.spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject = Subject(input=input, label=label)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0)
            label = output['label'].data.squeeze(0)

        # Get the OAR patch.
        # 'Parotid-Left' max extent in training data is (48.85mm, 61.52mm, 72.00mm), which equates
        # to a voxel shape of (48.85, 61.52, 24) for a spacing of (1mm, 1mm, 3mm).
        # Chose (128, 128, 96) for patch size as it's larger than the OAR extent and it fits into
        # the GPU memory.
        shape = (128, 128, 96)
        input, label = self._extract_patch(input, label, shape)

        # Determine result.
        result = (input, label)
        if self.raw_input:
            result += (input,)
        if self.raw_label:
            result += (label,)

        return result

    def _extract_patch(
        self,
        input: np.ndarray,
        label: np.ndarray,
        shape: Tuple[int, int, int]) -> np.ndarray:
        """
        returns: a patch around the OAR.
        args:
            input: the input data.
            label: the label data.
            shape: the shape of the patch. Must be larger than the extent of the OAR.
        """
        # Find OAR extent.
        non_zero = np.argwhere(label != 0)
        mins = non_zero.min(axis=0)
        maxs = non_zero.max(axis=0)
        oar_shape = maxs - mins

        # Determine min/max indices of the patch.
        shape_diff = shape - oar_shape
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = mins - lower_add
        maxs = mins + shape

        # Crop or pad the volume.
        input = self._crop_or_pad(input, mins, maxs, fill=input.min()) 
        label = self._crop_or_pad(label, mins, maxs)

        return input, label

    def _crop_or_pad(
        self,
        array: np.ndarray,
        mins: Tuple[int, int, int],
        maxs: Tuple[int, int, int],
        fill: Union[int, float] = 0) -> np.ndarray:
        """
        returns: extracts a patch from a 3D array, cropping/padding as necessary.
        args:
            array: the array data.
            mins: the minimum indices along each dimension. Negative indices
                indicate that padding should be added.
            maxs: the maximum indices along each dimension. Indices larger
                than the array shape indicate padding should be added.
        kwargs:
            fill: the padding value.
        """
        # Check for negative indices, and record padding.
        lower_pad = (-mins).clip(0) 
        mins = mins.clip(0)

        # Check for max indices larger than input, and record padding.
        upper_pad = (maxs - array.shape).clip(0)
        maxs = maxs - upper_pad

        # Perform crop.
        slices = tuple(slice(min, max) for min, max in zip(mins, maxs))
        array = array[slices]

        # Perform padding.
        padding = tuple(zip(lower_pad, upper_pad))
        array = np.pad(array, padding, constant_values=fill)

        return array

class ParotidLeft3DSegmenterVisualSampler(Sampler):
    def __init__(self, dataset, num_images, seed):
        self.dataset_length = len(dataset)
        self.num_images = num_images
        self.seed = seed

    def __iter__(self):
        # Set random seed for repeatability.
        np.random.seed(self.seed)

        # Get random subset of indices.
        indices = list(range(self.dataset_length))
        np.random.shuffle(indices)
        indices = indices[:self.num_images]

        return iter(indices)

    def __len__(self):
        return self.num_images
