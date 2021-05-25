import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchio import LabelMap, ScalarImage, Subject

from mymi import config

class ParotidLeft3DSegmenterLoader:
    @staticmethod
    def build(folder, batch_size=1, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        returns: a data loader.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
        kwargs:
            batch_size: the number of images in the batch.
            raw_input: return the non-transformed input also.
            raw_label: return the non-transformed label also.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DSegmenterDataset(folder, raw_input=raw_input, raw_label=raw_label, spacing=spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset)

class ParotidLeft3DSegmenterDataset(Dataset):
    def __init__(self, folder, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            spacing: the voxel spacing of the data.
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
        folder_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', folder)
        self.samples = np.reshape([os.path.join(folder_path, p) for p in sorted(os.listdir(folder_path))], (-1, 2))
        self.num_samples = len(self.samples)

    def __len__(self):
        """
        returns: number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(
        self,
        idx: int):
        """
        returns: an (input, label) pair from the dataset.
        args:
            idx: the index of the item to return.
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
            subject = Subject(one_image=input, a_segmentation=label)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input = output['one_image'].data.squeeze(0).numpy()
            label = output['a_segmentation'].data.squeeze(0).numpy()

        # Required extent.
        # From 'Segmenter dataloader' notebook, max extent in training data is (48.85mm, 61.52mm, 72.00mm).
        # Converting to voxel width we have: (48.85, 61.52, 24) for a spacing of (1.0mm, 1.0mm, 3.0mm).
        # We can choose a patch that is larger than the required voxel width, and that we know fits into the GPU
        # as we use it for the localiser training: (128, 128, 96), giving physical size of (128mm, 128mm, 288mm)
        # which is more than large enough. We can probably trim this later.
        required_shape = (128, 128, 96)

        # Find OAR extent.
        non_zero = np.argwhere(label != 0)
        mins = non_zero.min(axis=0)
        maxs = non_zero.max(axis=0)
        oar_shape = maxs - mins

        # Pad the OAR to the required shape.
        shape_diff = required_shape - oar_shape
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = mins - lower_add
        maxs = mins + required_shape

        # Crop or pad the volume.
        input = self._crop_or_pad(input, mins, maxs, fill=input.min()) 
        label = self._crop_or_pad(label, mins, maxs)

        # Determine result.
        result = (input, label)
        if self.raw_input:
            result += (input,)
        if self.raw_label:
            result += (label,)

        return result

    def _crop_or_pad(
        self,
        array: np.ndarray,
        mins: Tuple[int, int, int],
        maxs: Tuple[int, int, int],
        fill: Union[int, float] = 0) -> np.ndarray:
        """
        returns: a 3D array that has been cropped or padded according to limits.
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
