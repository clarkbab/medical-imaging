import numpy as np
from torch.utils import data

from ..loaders import ImageDataset

class DataLoader:
    @staticmethod
    def build(type, batch_size=32, transforms=[]):
        """
        returns: a data loader.
        type: a string describing the desired loader - 'train', 'validate' or 'test'.
        transforms: an array of augmentation transforms.
        """
        # Create dataset object.
        dataset = ImageDataset(type, transforms)

        # Create weighted sampler.
        pos_weights = np.ones(dataset.num_pos) / dataset.num_pos
        neg_weights = np.ones(dataset.num_neg) / dataset.num_neg
        weights = np.concatenate((pos_weights, neg_weights))
        sampler = data.WeightedRandomSampler(weights, len(weights), replacement=True)

        # Create loader.
        return data.DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)
