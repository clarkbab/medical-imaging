import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchio import LabelMap, ScalarImage, Subject

from mymi import config

class ParotidLeft3DLoader:
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
        dataset = ParotidLeft3DDataset(folder, raw_input=raw_input, raw_label=raw_label, spacing=spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset)

class ParotidLeft3DDataset(Dataset):
    def __init__(self, folder, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            spacing: the voxel spacing of the data.
        kwargs:
            raw_input: return the raw input data loaded from disk, in addition to transformed data.
            raw_label: return the raw label data loaded from disk, in addition to transformed data.
            spacing: the voxel spacing.
            transform: transformations to apply.
        """
        self.raw_input = raw_input
        self.raw_label = raw_label
        self.spacing = spacing
        self.transform = transform
        if transform:
            assert spacing, 'Spacing is required when transform applied to dataloader.'

        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        folder_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'training', 'parotid-left-3d', folder)
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
            input_t = np.expand_dims(input, axis=0)
            label_t = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1], 0, 0],
                [0, 0, self.spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input_t = ScalarImage(tensor=input_t, affine=affine)
            label_t = LabelMap(tensor=label_t, affine=affine)
            subject = Subject(one_image=input_t, a_segmentation=label_t)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input_t = output['one_image'].data.squeeze(0)
            label_t = output['a_segmentation'].data.squeeze(0)

        # Determine result.
        result = (input_t, label_t)
        if self.raw_input:
            result += (input,)
        if self.raw_label:
            result += (label,)

        return result
