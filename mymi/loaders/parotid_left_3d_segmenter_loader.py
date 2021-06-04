import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchio
from torchio import LabelMap, ScalarImage, Subject
from typing import *

from mymi import config

class ParotidLeft3DSegmenterLoader:
    @staticmethod
    def build(
        folder: str,
        patch_size: Tuple[int, int, int],
        batch_size: int = 1,
        p: float = 1,
        raw_input: bool = False,
        raw_label: bool = False,
        spacing: Tuple[float, float, float] = None,
        transform: torchio.transforms.Transform = None) -> torch.utils.data.DataLoader:
        """
        returns: a data loader.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            patch_size: the patch size to extract.
        kwargs:
            batch_size: the number of images in the batch.
            p: the proportion of samples that include the Parotid.
            raw_input: return the non-transformed input also.
            raw_label: return the non-transformed label also.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DSegmenterDataset(folder, patch_size, p=p, raw_input=raw_input, raw_label=raw_label, spacing=spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, shuffle=True)

class ParotidLeft3DSegmenterDataset(Dataset):
    def __init__(
        self,
        folder: str,
        patch_size: Tuple[int, int, int],
        p: float = 1,
        raw_input: bool = False,
        raw_label: bool = False,
        spacing: Tuple[int, int, int] = None,
        transform: torchio.transforms.Transform = None):
        """
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            patch_size: the size of the patch.
        kwargs:
            p: the proportion of samples that are centred on a foreground voxel.
            raw_input: return the raw input data loaded from disk, in addition to transformed data.
            raw_label: return the raw label data loaded from disk, in addition to transformed data.
            spacing: the voxel spacing of the data on disk.
            transform: transformations to apply.
        """
        self._p = p
        self._patch_size = patch_size
        self._raw_input = raw_input
        self._raw_label = raw_label
        self._spacing = spacing
        self._transform = transform
        if transform:
            assert spacing, 'Spacing is required when transform applied to dataloader.'

        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        folder_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', folder)
        self._samples = np.reshape([os.path.join(folder_path, p) for p in sorted(os.listdir(folder_path))], (-1, 2))
        self._num_samples = len(self._samples)

    def __len__(self):
        """
        returns: number of samples in the dataset.
        """
        return self._num_samples

    def __getitem__(
        self,
        idx: int):
        """
        returns: an (input, label) pair from the dataset.
        args:
            idx: the index of the item to return.
        """
        # Get data and label paths.
        input_path, label_path = self._samples[idx]

        # Load data and label.
        f = open(input_path, 'rb')
        input = np.load(f)
        f = open(label_path, 'rb')
        label = np.load(f)

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
            subject = Subject(input=input, label=label)

            # Transform the subject.
            output = self._transform(subject)

            # Extract results.
            input = output['input'].data.squeeze(0).numpy()
            label = output['label'].data.squeeze(0).numpy()

        # Roll the dice.
        if np.random.binomial(1, self._p):
            input, label = self._extract_foreground_patch(input, label, self._patch_size)
        else:
            input, label = self._extract_random_patch(input, label, self._patch_size)

        # Get the OAR patch.

        # Determine result.
        result = (input, label)
        if self._raw_input:
            result += (input,)
        if self._raw_label:
            result += (label,)

        return result

    def _extract_foreground_patch(
        self,
        input: np.ndarray,
        label: np.ndarray,
        size: Tuple[int, int, int]) -> np.ndarray:
        """
        returns: a patch around the OAR.
        args:
            input: the input data.
            label: the label data.
            size: the size of the patch. Must be larger than the extent of the OAR.
        """
        # Find foreground voxels.
        fg_voxels = np.argwhere(label != 0)
        
        # Choose randomly from the foreground voxels.
        fg_voxel_idx = np.random.choice(len(fg_voxels))
        centre_voxel = fg_voxels[fg_voxel_idx]

        # Determine min/max indices of the patch.
        shape_diff = np.array(size) - 1
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = centre_voxel - lower_add
        maxs = mins + size

        # Crop or pad the volume.
        input = self._crop_or_pad(input, mins, maxs, fill=input.min()) 
        label = self._crop_or_pad(label, mins, maxs)

        return input, label

    def _extract_random_patch(
        self,
        input: np.ndarray,
        label: np.ndarray,
        size: Tuple[int, int, int]) -> np.ndarray:
        """
        returns: a random patch from the volume.
        args:
            input: the input data.
            label: the label data.
            size: the size of the patch.
        """
        # Choose a random voxel.
        centre_voxel = tuple(map(np.random.randint, size))

        # Determine min/max indices of the patch.
        shape_diff = np.array(size) - 1
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = centre_voxel - lower_add
        maxs = mins + size

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
