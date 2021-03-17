import numpy as np
import os

FILENAME_NUM_DIGITS = 5

class ProcessedDataset:
    ###
    # Subclasses must implement.
    ###

    @classmethod
    def data_dir(cls):
        raise NotImplementedError("Method 'data_dir' not implemented in subclass.")

    ###
    # Basic queries.
    ###

    @classmethod
    def input(cls, folder, sample_idx):
        """
        returns: the input data for sample i.
        args:
            folder: 'train', 'test' or 'validate'.
            sample_idx: the sample index to load.
        """
        # Load the input data.
        filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-input"
        input_path = os.path.join(cls.data_dir(), folder, filename)
        f = open(input_path, 'rb')
        input = np.load(f)

        return input

    @classmethod
    def label(cls, folder, sample_idx):
        """
        returns: the label data for sample i.
        args:
            folder: 'train', 'test' or 'validate'.
            sample_idx: the sample index to load.
        """
        # Load the label data.
        filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-label"
        label_path = os.path.join(cls.data_dir(), folder, filename)
        f = open(label_path, 'rb')
        label = np.load(f)

        return label

    @classmethod
    def sample(cls, folder, sample_idx):
        """
        returns: the (input, label) pair for the given sample index.
        args:
            folder: 'train', 'test' or 'validate'.
            sample_idx: the sample index to load.
        """
        return cls.input(folder, sample_idx), cls.label(folder, sample_idx)
