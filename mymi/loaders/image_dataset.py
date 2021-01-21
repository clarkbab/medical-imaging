import numpy as np
import os
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, type, data_dir, transforms=[]):
        """
        verbose: print information.
        type: train, validate or test.
        """
        self.root_dir = os.path.join(data_dir, type)
        self.pos_dir = os.path.join(self.root_dir, 'positive')
        self.neg_dir = os.path.join(self.root_dir, 'negative')
        pos_samples = np.reshape([os.path.join(self.pos_dir, p) for p in sorted(os.listdir(self.pos_dir))], (-1, 2))
        neg_samples = np.reshape([os.path.join(self.neg_dir, p) for p in sorted(os.listdir(self.neg_dir))], (-1, 2))
        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
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
            input, label = transform(input, label)

        return input, label