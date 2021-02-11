import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

data_path = os.environ['MYMI_DATA']
DATA_DIR = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')

class ParotidLeft3DLoader:
    @staticmethod
    def build(folder, batch_size=32, data_dir=DATA_DIR, transforms=[]):
        """
        returns: a data loader.
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
        kwargs:
            batch_size: the number of images in the batch.
            data_dir: the location of the image data.
            transforms: an array of augmentation transforms.
        """
        # Create dataset object.
        dataset = ParotidLeft3DDataset(folder, data_dir, transforms=transforms)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset)

class ParotidLeft3DDataset(Dataset):
    def __init__(self, folder, data_dir, transforms=[]):
        """
        args:
            folder: a string describing the desired loader - 'train', 'validate' or 'test'.
            data_dir: the location of the data.
        kwargs:
            transforms: a list of transforms to apply.
        """
        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        self.root_dir = os.path.join(data_dir, folder)
        self.samples = np.reshape([os.path.join(self.root_dir, p) for p in sorted(os.listdir(self.root_dir))], (-1, 2))
        self.num_samples = len(self.samples)
        self.transforms = transforms

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

        # Perform transforms.
        logging.info(f"Transforming sample: {idx}.")
        for transform in self.transforms:
            logging.info(f"Transforms: {transform}.")
            # Get deterministic transform.
            det_transform = transform.deterministic()

            # Apply to input and label.
            input = det_transform(input)
            label = det_transform(label, binary=True)
        logging.info('Finished transform.')

        return input, label
