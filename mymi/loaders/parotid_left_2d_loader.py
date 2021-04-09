import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from mymi import config

class ParotidLeft2DLoader:
    @staticmethod
    def build(folder, batch_size=32, transforms=[]):
        """
        returns: a data loader.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
        kwargs:
            batch_size: the number of images in the batch.
            transforms: an array of augmentation transforms.
        """
        # Create dataset object.
        dataset = ParotidLeft2DDataset(folder, transforms=transforms)

        # Create weighted sampler that draws 50/50 pos/neg samples.
        pos_weights = np.ones(dataset.num_pos) / dataset.num_pos
        neg_weights = np.ones(dataset.num_neg) / dataset.num_neg
        weights = np.concatenate((pos_weights, neg_weights))
        num_samples = 2 * dataset.num_pos           # There are more pos than neg samples.
        sampler = WeightedRandomSampler(weights, num_samples)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)

class ParotidLeft2DDataset(Dataset):
    def __init__(self, folder, transforms=[]):
        """
        verbose: print information.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
        kwargs:
            transforms: a list of transforms to apply.
        """
        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        self.destination_dir = os.path.join(config.dataset_dir, 'datasets', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-2d', folder)
        self.pos_dir = os.path.join(self.destination_dir, 'positive')
        self.neg_dir = os.path.join(self.destination_dir, 'negative')
        pos_samples = np.reshape([os.path.join(self.pos_dir, p) for p in sorted(os.listdir(self.pos_dir))], (-1, 2))
        neg_samples = np.reshape([os.path.join(self.neg_dir, p) for p in sorted(os.listdir(self.neg_dir))], (-1, 2))
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)

        # Save paths to samples. We'll index with positive samples first.
        self.samples = np.concatenate((pos_samples, neg_samples))

        self.transforms = transforms

    def __len__(self):
        """
        returns: number of samples in the dataset.
        """
        return self.num_pos + self.num_neg

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

        # Perform transforms.
        for transform in self.transforms:
            input = transform(input)
            label = transform
            input, label = transform(input, label)

        return input, label
