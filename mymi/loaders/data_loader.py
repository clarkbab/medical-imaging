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
        p: the proportion of positive samples.
        """
        # Create dataset object.
        dataset = ImageDataset(type, transforms)

        # Create weighted sampler that draws 50/50 pos/neg samples.
        pos_weights = np.ones(dataset.num_pos) / dataset.num_pos
        neg_weights = np.ones(dataset.num_neg) / dataset.num_neg
        weights = np.concatenate((pos_weights, neg_weights))
        num_samples = 2 * dataset.num_pos           # There are more pos than neg samples.
        sampler = data.WeightedRandomSampler(weights, num_samples)

        # Create loader.
        return data.DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)
