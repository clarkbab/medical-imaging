import numpy as np
import os
from torch.utils import data

from ..loaders import NegativeImageDataset

DATA_DIR_DEFAULT = os.path.join(os.sep, 'media', 'brett', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '2d-parotid-left')

class NegativeSampler(data.Sampler):
    def __init__(self, dataset, num_images, seed=42):
        self.dataset_length = len(dataset)
        self.num_images = num_images
        self.seed = seed

    def __iter__(self):
        indices = list(range(self.dataset_length))
        np.random.seed(self.seed)           # Same indices should be returned each epoch.
        np.random.shuffle(indices)
        indices = indices[:self.num_images]
        return iter(indices)

    def __len__(self):
        return self.num_images

class NegativeLoader:
    @staticmethod
    def build(batch_size=32, transforms=[], data_dir=DATA_DIR_DEFAULT):
        """
        returns: a data loader.
        transforms: an array of augmentation transforms.
        """
        # Create dataset object.
        dataset = NegativeImageDataset('validate', data_dir, transforms=transforms)

        # Create sampler.
        sampler = NegativeSampler(dataset, 5 * batch_size)

        # Create loader.
        return data.DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)
