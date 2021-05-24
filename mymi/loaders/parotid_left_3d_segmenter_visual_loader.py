import logging
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, Sampler
from torchio import LabelMap, ScalarImage, Subject

from mymi import config

class ParotidLeft3DSegmenterVisualLoader:
    @staticmethod
    def build(num_batches=5, seed=42, batch_size=1, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        returns: a data loader.
        kwargs:
            batch_size: the number of images in the batch.
            num_batches: how many batches this loader should generate.
            seed: random number generator seed.
            spacing: the voxel spacing of the data.
            transform: the transform to apply.
        """
        # Create dataset object.
        dataset = ParotidLeft3DSegmenterVisualDataset(raw_input=raw_input, raw_label=raw_label, spacing=spacing, transform=transform)

        # Create sampler.
        sampler = ParotidLeft3DSegmenterVisualSampler(dataset, num_batches * batch_size, seed)

        # Create loader.
        return DataLoader(batch_size=batch_size, dataset=dataset, sampler=sampler)

class ParotidLeft3DSegmenterVisualDataset(Dataset):
    def __init__(self, raw_input=False, raw_label=False, spacing=None, transform=None):
        """
        kwargs:
            raw_input: return the raw input data loaded from disk, in addition to transformed data.
            raw_label: return the raw label data loaded from disk, in addition to transformed data.
            spacing: the voxel spacing of the data on disk.
            transform: transformations to apply.
        """
        self.raw_input = raw_input
        self.raw_label = raw_label
        self.spacing = spacing
        self.transform = transform
        if transform:
            assert spacing, 'Spacing is required when transform applied to dataloader.'

        # Load up samples into 2D arrays of (input_path, label_path) pairs.
        folder_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'validate')
        self.samples = np.reshape([os.path.join(folder_path, p) for p in sorted(os.listdir(folder_path))], (-1, 2))
        self.num_samples = len(self.samples)

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
                [self.spacing[0], 0, 0, 0],
                [0, self.spacing[1], 0, 0],
                [0, 0, self.spacing[2], 1],
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

        # Required extent.
        # From 'Segmenter dataloader' notebook, max extent in training data is (48.85mm, 61.52mm, 72.00mm).
        # Converting to voxel width we have: (48.85, 61.52, 24) for a spacing of (1.0mm, 1.0mm, 3.0mm).
        # We can choose a patch that is larger than the required voxel width, and that we know fits into the GPU
        # as we use it for the localiser training: (128, 128, 96), giving physical size of (128mm, 128mm, 288mm)
        # which is more than large enough. We can probably trim this later.
        extent = (128, 128, 96)

        # Find OAR extent.
        non_zero = np.argwhere(label != 0)
        mins = non_zero.min(axis=0)
        maxs = non_zero.max(axis=0)
        voxel_widths = maxs - mins

        # Pad the OAR, preferencing lower indices.
        to_add = extent - voxel_widths
        half_add = np.ceil(to_add / 2).astype(int)
        min_voxels = mins - half_add

        # Extract patch.
        slices = tuple(slice(m, m + w) for m, w in zip(min_voxels, extent))
        input = input[slices]
        label = label[slices]

        # Determine result.
        result = (input, label)
        if self.raw_input:
            result += (input,)
        if self.raw_label:
            result += (label,)

        return result

class ParotidLeft3DSegmenterVisualSampler(Sampler):
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
