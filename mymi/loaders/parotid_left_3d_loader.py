import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchio import LabelMap, ScalarImage, Subject

data_path = os.environ['MYMI_DATA']
DATA_DIR = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')

class ParotidLeft3DLoader:
    @staticmethod
    def build(folder, batch_size=32, data_dir=DATA_DIR, transform=None):
        """
        returns: a data loader.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
        kwargs:
            batch_size: the number of images in the batch.
            data_dir: the location of the image data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DDataset(folder, data_dir, transform=transform)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset)

class ParotidLeft3DDataset(Dataset):
    def __init__(self, folder, data_dir, input_only_transform=None, transform=None):
        """
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            data_dir: the location of the data.
        kwargs:
            input_only_transform: transformation for the input only.
            transform: transformations to apply.
        """
        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        self.root_dir = os.path.join(data_dir, folder)
        self.samples = np.reshape([os.path.join(self.root_dir, p) for p in sorted(os.listdir(self.root_dir))], (-1, 2))
        self.num_samples = len(self.samples)
        self.transform = transform

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
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 3, 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject = Subject(one_image=input, a_segmentation=label)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input = output['one_image'].data.squeeze(0)
            label = output['a_segmentation'].data.squeeze(0)

        return input, label
