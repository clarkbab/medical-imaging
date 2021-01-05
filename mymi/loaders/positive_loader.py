import numpy as np
from torch.utils import data

from ..loaders import PositiveImageDataset

class PositiveSampler(data.Sampler):
    def __init__(self, dataset, num_images, seed=42):
        self.dataset_length = len(dataset)
        self.num_images = num_images
        self.seed = seed

    def __iter__(self):
        indices = list(range(self.dataset_length))
        np.random.seed(self.seed)           # Same indices should be returned each epoch.
        np.random.shuffle(indices)
        indices = indices[:self.num_images]
        print(f"Positive indices: {indices}")
        return iter(indices)

    def __len__(self):
        return self.num_images

class PositiveLoader:
    @staticmethod
    def build(batch_size=32, transforms=[]):
        """
        returns: a data loader.
        transforms: an array of augmentation transforms.
        """
        # Create dataset object.
        dataset = PositiveImageDataset('validate', transforms)

        # Create sampler.
        sampler = PositiveSampler(dataset, 5 * batch_size)

        # Create loader.
        return data.DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)
