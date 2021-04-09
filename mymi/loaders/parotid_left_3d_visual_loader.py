import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from torchio import Compose, LabelMap, ScalarImage, Subject

from mymi import config

class ParotidLeft3DVisualLoader:
    @staticmethod
    def build(batch_size=32, num_batches=5, seed=42, transform=None):
        """
        returns: a data loader.
        kwargs:
            batch_size: the number of images in a batch.
            num_batches: how many batches this loader should generate.
            seed: random number generator seed.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DVisualDataset(transform=transform)

        # Create sampler.
        sampler = ParotidLeft3DVisualSampler(dataset, num_batches * batch_size, seed)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)

class ParotidLeft3DVisualDataset(Dataset):
    def __init__(self, transform=None):
        """
        returns: a dataset.
        kwargs:
            transforms: an array of augmentation transforms.
        """
        # Load paths to all samples.
        self.data_dir = os.path.join(config.dataset_dir, 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d', 'validate')
        samples = np.reshape([os.path.join(self.data_dir, p) for p in sorted(os.listdir(self.data_dir))], (-1, 2))

        self.num_samples = len(samples)
        self.samples = samples
        self.transform = transform

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

        # Perform transform.
        if self.transform:
            # Add 'batch' dimension.
            input = np.expand_dims(input, axis=0)
            label = np.expand_dims(label, axis=0)

            # Create 'subject'.
            affine = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 3, 1],
                [0, 0, 0, 1]
            ])
            input = ScalarImage(tensor=input, affine=affine)
            label = LabelMap(tensor=label, affine=affine)
            subject = Subject(one_image=input, a_segmentation=label)

            # Transform the subject.
            output = self.transform(subject)

            # Extract results.
            input = output['one_image'].data.squeeze(0)
            label = output['a_segmentation'].data.squeeze(0)

        return input, label

class ParotidLeft3DVisualSampler(Sampler):
    def __init__(self, dataset, num_images, seed):
        self.dataset_length = len(dataset)
        self.num_images = num_images
        self.seed = seed

    def __iter__(self):
        # Set random seed for repeatability.
        np.random.seed(self.seed)

        # Get random subset of indices.
        indices = list(range(self.dataset_length))
        np.random.shuffle(indices)
        indices = indices[:self.num_images]

        return iter(indices)

    def __len__(self):
        return self.num_images
