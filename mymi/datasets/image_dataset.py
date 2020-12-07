import numpy as np
import os
from torch.utils.data import Dataset

ROOT_DIR = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '2d-parotid-left', 'Parotid-Left')
POS_DIR = os.path.join(ROOT_DIR, 'positive')
NEG_DIR = os.path.join(ROOT_DIR, 'negative')

class ImageDataset(Dataset):
    def __init__(self, verbose=False, transforms=[]):
        """
        verbose: print information.
        """
        self.num_pos = int(len(os.listdir(POS_DIR)) / 2)
        self.num_neg = int(len(os.listdir(NEG_DIR)) / 2)
        self.transforms = transforms
        self.verbose = verbose

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
        if self.verbose:
            print(f"Loading sample '{idx}'.")

        # Get directory and sub-index.
        data_dir = POS_DIR if idx < self.num_pos else NEG_DIR
        sub_idx = idx if idx < self.num_pos else idx - self.num_pos

        # Get data and label paths.
        input_path = os.path.join(data_dir, f"{sub_idx:05}-input") 
        label_path = os.path.join(data_dir, f"{sub_idx:05}-label")

        # Load data and label.
        f = open(input_path, 'rb')
        input = np.load(f)
        f = open(label_path, 'rb')
        label = np.load(f)

        # Perform transforms.
        for transform in self.transforms:
            input, label = transform(input, label)

        return input, label