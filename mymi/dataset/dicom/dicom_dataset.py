from collections import OrderedDict
import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import os
import pandas as pd
from pandas import DataFrame
import pydicom as dicom
from scipy.ndimage import center_of_mass
from skimage.draw import polygon
from torchio import ScalarImage, Subject
from tqdm import tqdm
from typing import *

from mymi import cache
from mymi.cache import cached_method
from mymi import config
from mymi.utils import filterOnNumPats, filterOnPatIDs

from .hierarchical import require_hierarchical
from .patient import Patient

Z_SPACING_ROUND_DP = 2

class DicomDataset:
    def __init__(
        self,
        name: str):
        """
        args:
            name: the name of the dataset.
        """
        self._name = name
        self._path = os.path.join(config.directories.datasets, name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @require_hierarchical
    def list_patients(self) -> Sequence[str]:
        """
        returns: a list of patient IDs.
        """
        # Return top-level folders from 'hierarchical' dataset.
        hier_path = os.path.join(self._path, 'hierarchical')
        return list(sorted(os.listdir(hier_path)))

    @require_hierarchical
    def patient(
        self,
        id: str) -> Patient:
        """
        returns: a Patient object.
        args:
            id: the patient ID.
        """
        # Check that patient ID exists.
        pat_path = os.path.join(self._path, 'hierarchical', id)
        if not os.path.isdir(pat_path):
            raise ValueError(f"Patient '{id}' not found in dataset '{self._name}'.")

        # Create patient.
        pat = Patient(self._name, id)

        return pat

    @require_hierarchical
    @cached_method('_name')
    def ct_distribution(
        self, 
        bin_width: int = 10,
        clear_cache: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> OrderedDict:
        """
        effect: plots CT distribution of the dataset.
        kwargs:
            bin_width: the width of the histogram bins.
            clear_cache: forces the cache to clear.
            labels: include patients with any of the listed labels (behaves like an OR).
            num_pats: the number of patients to include.
            pat_ids: the patients to include.
        """
        # Load all patients.
        pats = self.list_patients()
        
        # Filter patients.
        pats = list(filter(self.filterOnPatIDs(pat_ids), pats))
        pats = list(filter(filterOnLabels(labels), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Calculate the frequencies.
        freqs = {}
        for pat in tqdm(pats):
            # Load patient volume.
            ct_data = self.patient(pat).ct_data(clear_cache=clear_cache)

            # Bin the data.
            binned_data = bin_width * np.floor(ct_data / bin_width)

            # Get values and their frequencies.
            values, frequencies = np.unique(binned_data, return_counts=True)

            # Add values to frequencies dict.
            for v, f in zip(values, frequencies):
                # Check if value has already been added.
                if v in freqs:
                    freqs[v] += f
                else:
                    freqs[v] = f

        # Fill in empty bins.
        values = np.fromiter(freqs.keys(), dtype=np.float)
        min, max = values.min(), values.max() + bin_width
        bins = np.arange(min, max, bin_width)
        for b in bins:
            if b not in freqs:
                freqs[b] = 0            

        # Sort dictionary.
        freqs = OrderedDict(sorted(freqs.items()))

        return freqs

    @require_hierarchical
    @cached_method('_name')
    def ct_summary(
        self, 
        clear_cache: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> DataFrame:
        """
        returns: a DataFrame with patient CT summaries.
        kwargs:
            clear_cache: force the cache to clear.
            labels: include patients with (at least) on of the labels.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
        """
        # Define table structure.
        cols = {
            'fov-x': float,
            'fov-y': float,
            'fov-z': float,
            'hu-max': float,
            'hu-min': float,
            'num-missing': int,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
            'pat-id': str,
            'size-x': int,
            'size-y': int,
            'size-z': int,
            'spacing-x': float,
            'spacing-y': float,
            'spacing-z': float,
        }
        df = pd.DataFrame(columns=cols.keys())

        # List patients.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(self.filterOnPatIDs(pat_ids), pats))
        pats = list(filter(filterOnLabels(labels), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Add patient info.
        for pat in tqdm(pats):
            pat_df = self.patient(pat).ct_summary(clear_cache=clear_cache)
            pat_df['pat-id'] = pat
            df = df.append(pat_df)

        # Set column type.
        df = df.astype(cols)
        
        # Set index.
        df = df.set_index('pat-id')

        return df

    @require_hierarchical
    @cached_method('_name')
    def label_summary(
        self, 
        clear_cache: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> DataFrame:
        """
        returns: a DataFrame with patient labels and information.
        kwargs:
            clear_cache: force the cache to clear.
            labels: include patients with (at least) on of the labels.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
        """
        # Define table structure.
        cols = {
            'patient-id': 'object',
            'label': 'object',
            'com-x': np.uint16,
            'com-y': np.uint16,
            'com-z': np.uint16,
            'width-x': np.uint16,
            'width-y': np.uint16,
            'width-z': np.uint16
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(self.filterOnPatIDs(pat_ids), pats))
        pats = list(filter(filterOnLabels(labels), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Add patient labels.
        for pat in tqdm(pats):
            summary_df = self.patient(pat).label_summary(clear_cache=clear_cache, labels=labels)

            # Add rows.
            for _, row in summary_df.iterrows():
                data = {
                    'patient-id': pat,
                    'label': row['label'],
                    'com-x': row['com-x'],
                    'com-y': row['com-y'],
                    'com-z': row['com-z'],
                    'width-x': row['width-x'],
                    'width-y': row['width-y'],
                    'width-z': row['width-z']
                }
                df = df.append(data, ignore_index=True)

        return df

    @classmethod
    def ct_statistics(cls, label='all'):
        """
        returns: a dataframe of CT statistics for the entire dataset.
        kwargs:
            label: only include data for patients with the label.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'kwargs': {
                'label': label
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result

        # Convert 'labels'.
        if isinstance(label, str) and label != 'all':
            label = [label]

        # Define dataframe structure.
        cols = {
            'hu-mean': 'float64',
            'hu-std-dev': 'float64'
        }
        df = pd.DataFrame(columns=cols.keys())

        # Get patients IDs.
        pat_ids = cls.list_patients()

        # Calculate mean.
        total = 0
        num_voxels = 0
        for pat_id in pat_ids:
            # Get patient labels.
            pat_labels = list(cls.patient_labels(pat_id)['label'])
            
            # Skip if patient has no labels, or doesn't have the specified labels.
            if len(pat_labels) == 0 or (label != 'all' and not np.array_equal(np.intersect1d(label, pat_labels), label)):
                continue

            # Add data for this patient.
            data = cls.patient_data(pat_id)
            total += data.sum()
            num_voxels += data.shape[0] * data.shape[1] * data.shape[2]
        mean = total / num_voxels

        # Calculate standard dev.
        total = 0
        for pat_id in pat_ids:
            # Add data for the patient.
            data = cls.patient_data(pat_id)
            total += ((data - mean) ** 2).sum()
        std_dev = np.sqrt(total / num_voxels)

        # Add data.
        data = {
            'hu-mean': mean,
            'hu-std-dev': std_dev
        }
        df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    def filterOnLabels(
        self,
        labels: Union[str, Sequence[str]]) -> Callable[[str], bool]:
        """
        returns: a function to filter patients on whether they have that labels.
        args:
            labels: the passed 'labels' kwarg.
        """
        def fn(id):
            # Load patient labels.
            pat_labels = self.patient(id).label_summary().label.to_numpy()

            if ((isinstance(labels, str) and (labels == 'all' or labels in pat_labels)) or
                ((isinstance(labels, list) or isinstance(labels, ndarray) or isinstance(labels, tuple)) and len(np.intersect1d(labels, pat_labels)) != 0)):
                return True
            else:
                return False

        return fn
