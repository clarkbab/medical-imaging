import numpy as np
import os
from torch.utils.data import Dataset

ROOT_DIR = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '2d-parotid-left', 'Parotid-Left')
POS_DIR = os.path.join(ROOT_DIR, 'positive')
NEG_DIR = os.path.join(ROOT_DIR, 'negative')

class ImageDataset(Dataset):
    def __init__(self):
        self.num_pos = len(os.listdir(POS_DIR))
        self.num_neg = len(os.listdir(NEG_DIR))

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
        # Get directory and sub-index.
        data_dir = POS_DIR if idx < self.num_pos else NEG_DIR
        sub_idx = idx if idx < self.num_pos else idx - self.num_pos

        # Get data and label paths.
        input_path = os.path.join(dir, f"{sub_idx}-input") 
        label_path = os.path.join(dir, f"{sub_idx}-label")

        # Load data and label.
        f = open(input_path, 'rb')
        data = np.load(f)
        f = open(label_path, 'rb')
        label = np.load(f)

        return data, label