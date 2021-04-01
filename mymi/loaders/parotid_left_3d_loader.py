import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchio import LabelMap, ScalarImage, Subject

data_path = os.environ['MYMI_DATA']
DATA_DIR = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')

class ParotidLeft3DLoader:
    @staticmethod
    def build(folder, batch_size=1, spacing=(1, 1, 3), transform=None):
        """
        returns: a data loader.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
        kwargs:
            batch_size: the number of images in the batch.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DDataset(folder, spacing, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset)

class ParotidLeft3DDataset(Dataset):
    def __init__(self, folder, spacing, transform=None):
        """
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            spacing: the voxel spacing of the data.
        kwargs:
            transform: transformations to apply.
        """
        self.transform = transform
        self.spacing = spacing

        # Labels shouldn't be transformed when evaluating.
        self.transform_label = False if folder == 'test' else True

        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        folder_path = os.path.join(DATA_DIR, folder)
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
            if self.transform_label:
                label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1], 0, 0],
                [0, 0, self.spacing[2], 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            if self.transform_label:
                label = LabelMap(tensor=label, affine=affine)
                subject = Subject(one_image=input, a_segmentation=label)
            else:
                subject = Subject(one_image=input)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input = output['one_image'].data.squeeze(0)
            if self.transform_label:
                label = output['a_segmentation'].data.squeeze(0)

        return input, label
