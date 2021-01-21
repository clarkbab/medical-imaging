import numpy as np
import os
from torch.utils import data

from ..loaders import ImageDataset

DATA_DIR_DEFAULT = os.path.join(os.sep, 'media', 'brett', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '2d-parotid-left')

class DataLoader:
    @staticmethod
    def build(type, batch_size=32, transforms=[], data_dir=DATA_DIR_DEFAULT):
        """
        returns: a data loader.
        type: a string describing the desired loader - 'train', 'validate' or 'test'.
        transforms: an array of augmentation transforms.
        p: the proportion of positive samples.
        """
        # Create dataset object.
        dataset = ImageDataset(type, data_dir, transforms=transforms)

        # Create weighted sampler that draws 50/50 pos/neg samples.
        pos_weights = np.ones(dataset.num_pos) / dataset.num_pos
        neg_weights = np.ones(dataset.num_neg) / dataset.num_neg
        weights = np.concatenate((pos_weights, neg_weights))
        num_samples = 2 * dataset.num_pos           # There are more pos than neg samples.
        sampler = data.WeightedRandomSampler(weights, num_samples)

        # Create loader.
        return data.DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)
