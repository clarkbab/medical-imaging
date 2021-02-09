import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Sampler

DATA_DIR_DEFAULT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')

class ParotidLeft3DSubsetLoader:
    @staticmethod
    def build(batch_size=32, data_dir=DATA_DIR_DEFAULT, num_batches=5, seed=42, transforms=[]):
        """
        returns: a data loader.
        kwargs:
            batch_size: the number of images in a batch.
            num_batches: how many batches this loader should generate.
            data_dir: the location of the data.
            seed: random number generator seed.
            transforms: an array of augmentation transforms.
        """
        # Create dataset object.
        dataset = ParotidLeft3DSubsetDataset(data_dir, transforms=transforms)

        # Create sampler.
        sampler = ParotidLeft3DSubsetSampler(dataset, num_batches * batch_size, seed)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)

class ParotidLeft3DSubsetDataset(Dataset):
    def __init__(self, data_dir, transforms=[]):
        """
        returns: a dataset.
        args:
            data_dir: the location of the data.
        kwargs:
            transforms: an array of augmentation transforms.
        """
        # Load paths to all samples.
        self.data_dir = os.path.join(data_dir, 'validate')
        samples = np.reshape([os.path.join(self.data_dir, p) for p in sorted(os.listdir(self.data_dir))], (-1, 2))

        self.num_samples = len(samples)
        self.samples = samples
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
        for transform in self.transforms:
            # Get deterministic transform.
            det_transform = transform.deterministic()

            # Apply transforms to data.
            input = det_transform(input)
            label = det_transform(label, binary=True)

        return input, label

class ParotidLeft3DSubsetSampler(Sampler):
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
