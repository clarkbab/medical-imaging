from collections import OrderedDict
import inspect
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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

from .dicom_patient import DicomPatient

Z_SPACING_ROUND_DP = 2

class DicomDataset:
    def __init__(
        self,
        name: str,
        ct_from: str = None):
        """
        args:
            name: the name of the dataset.
        kwargs:
            ct_from: pull CT info from another dataset.
        """
        self._ct_from = ct_from
        self._name = name
        self._path = os.path.join(config.directories.datasets, name)

        # Check if datasets exist.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{name}' not found.")
        if ct_from:
            ct_path = os.path.join(config.directories.datasets, ct_from)
            if not os.path.exists(ct_path):
                raise ValueError(f"Dataset '{ct_from}' not found.")

    @property
    def ct_from(self) -> str:
        return self._ct_from

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    def label_map(
        self,
        dataset: str = None) -> pd.DataFrame:
        """
        returns: a pd.DataFrame mapping internal region names to this dataset.
        kwargs:
            dataset: the other dataset, or internal representation if None.
        """
        # Load map file.
        filepath = os.path.join(self._path, 'label-map.csv')
        df = pd.read_csv(filepath)
        
        if dataset:
            # Load other map file.
            ds = DicomDataset(dataset)
            other_df = ds.label_map()

            # Merge the two maps.
            df = pd.merge(df, other_df, on='internal')
            df = df.rename(columns={ 'dataset_x': self._name, 'dataset_y': dataset })

        # Sort by internal name.
        df = df.sort_values('internal').reset_index(drop=True)

        return df

    def _require_hierarchical(fn: Callable) -> Callable:
        """
        effect: returns a wrapped function, ensuring hierarchical data has been built.
        args:
            fn: the wrapped function.
        """
        def wrapper(self, *args, **kwargs):
            if not self._hierarchical_exists():
                self._build_hierarchical()
            return fn(self, *args, **kwargs)
        return wrapper

    @_require_hierarchical
    def list_patients(self) -> Sequence[str]:
        """
        returns: a list of patient IDs.
        """
        # Return top-level folders from 'hierarchical' dataset.
        hier_path = os.path.join(self._path, 'hierarchical')
        return list(sorted(os.listdir(hier_path)))

    @_require_hierarchical
    def patient(
        self,
        id: Union[str, int]) -> DicomPatient:
        """
        returns: a DicomPatient object.
        args:
            id: the patient ID.
        """
        # Convert to string.
        if type(id) == int:
            id = str(id)

        # Check that patient ID exists.
        pat_path = os.path.join(self._path, 'hierarchical', id)
        if not os.path.isdir(pat_path):
            raise ValueError(f"Patient '{id}' not found in dataset '{self._name}'.")

        # Create patient.
        pat = DicomPatient(self._name, id, ct_from=self._ct_from)

        return pat

    @_require_hierarchical
    @cached_method('_ct_from', '_name')
    def info(
        self, 
        clear_cache: bool = False,
        filter_errors: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame with patient info.
        kwargs:
            clear_cache: force the cache to clear.
            filter_errors: exclude patients that produce known errors.
            labels: include patients with (at least) on of the labels.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
        """
        # Define table structure.
        cols = {
            'name': str,
            'patient-id': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(filterOnPatIDs(pat_ids), pats))
        pats = list(filter(self._filterOnLabels(labels, clear_cache=clear_cache, filter_errors=filter_errors), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Add patient labels.
        for pat in tqdm(pats):
            info_df = self.patient(pat).info(clear_cache=clear_cache)
            info_df['patient-id'] = pat
            df = df.append(info_df)

        # Set column type.
        df = df.astype(cols)

        return df

    @_require_hierarchical
    @cached_method('_ct_from', '_name')
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
        pats = list(filter(filterOnPatIDs(pat_ids), pats))
        pats = list(filter(self._filterOnLabels(labels), pats))
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

    @_require_hierarchical
    @cached_method('_ct_from', '_name')
    def ct_summary(
        self, 
        clear_cache: bool = False,
        filter_errors: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame with patient CT summaries.
        kwargs:
            clear_cache: force the cache to clear.
            filter_errors: exclude patients that produce known errors.
            labels: include patients with (at least) on of the labels.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'fov-x': float,
            'fov-y': float,
            'fov-z': float,
            'hu-max': float,
            'hu-min': float,
            'num-missing': int,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
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
        pats = list(filter(filterOnPatIDs(pat_ids), pats))
        pats = list(filter(self._filterOnLabels(labels, filter_errors=filter_errors), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Add patient info.
        for pat in tqdm(pats):
            # Load patient CT info.
            try:
                pat_df = self.patient(pat).ct_summary(clear_cache=clear_cache)
            except ValueError as e:
                if filter_errors:
                    logging.error(f"Patient filtered due to error calling 'ct_summary' for dataset '{self._name}', patient '{pat}'.")
                    logging.error(f"Filtered error: {e}")
                    continue
                else:
                    raise e

            # Add row.
            pat_df['patient-id'] = pat
            df = df.append(pat_df)

        # Set column type.
        df = df.astype(cols)

        return df

    @_require_hierarchical
    @cached_method('_name')
    def label_names(
        self,
        clear_cache: bool = False,
        filter_errors: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame with patient label names.
        kwargs:
            clear_cache: force the cache to clear.
            filter_errors: exclude patients that produce known errors.
            labels: include patients with (at least) on of the labels.
            num_pats: the number of patients to include.
            pat_ids: include listed patients.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'label': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(filterOnPatIDs(pat_ids), pats))
        pats = list(filter(self._filterOnLabels(labels, clear_cache=clear_cache, filter_errors=filter_errors), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Add patient labels.
        for pat in tqdm(pats):
            # Load patient labels.
            label_names_df = self.patient(pat).label_names(clear_cache=clear_cache)

            # Add rows.
            for _, row in label_names_df.iterrows():
                data = {
                    'patient-id': pat,
                    'label': row.label,
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    @_require_hierarchical
    @cached_method('_ct_from', '_name')
    def label_summary(
        self, 
        clear_cache: bool = False,
        filter_errors: bool = False,
        labels: Union[str, Sequence[str]] = 'all',
        num_pats: Union[str, int] = 'all',
        pat_ids: Union[str, Sequence[str]] = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame with patient labels and information.
        kwargs:
            clear_cache: force the cache to clear.
            filter_errors: exclude patients that produce known errors.
            labels: include patients with (at least) on of the labels.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'label': str,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(filterOnPatIDs(pat_ids), pats))
        pats = list(filter(self._filterOnLabels(labels, clear_cache=clear_cache, filter_errors=filter_errors), pats))
        pats = list(filter(filterOnNumPats(num_pats), pats))

        # Add patient labels.
        for pat in tqdm(pats):
            # Load patient summary.
            try:
                summary_df = self.patient(pat).label_summary(clear_cache=clear_cache, labels=labels)
            except ValueError as e:
                if filter_errors:
                    logging.error(f"Patient filtered due to error calling 'ct_summary' for dataset '{self._name}', patient '{pat}'.")
                    logging.error(f"Filtered error: {e}")
                    continue
                else:
                    raise e

            # Add rows.
            for _, row in summary_df.iterrows():
                data = {
                    'patient-id': pat,
                    'label': row.label,
                    'width-mm-x': row['width-mm-x'],
                    'width-mm-y': row['width-mm-y'],
                    'width-mm-z': row['width-mm-z']
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

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
            'hu-mean': float,
            'hu-std-dev': float,
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

    def _filterOnLabels(
        self,
        labels: Union[str, Sequence[str]],
        clear_cache: bool = False,
        filter_errors: bool = False) -> Callable[[str], bool]:
        """
        returns: a function that filters patient on label presence.
        args:
            clear_cache: force the cache to clear.
            labels: the passed 'labels' kwarg.
            filter_errors: exclude patients that produce known errors.
        """
        def fn(id):
            # Load patient labels.
            try:
                pat_labels = self.patient(id).label_names(clear_cache=clear_cache).label.to_numpy()
            except ValueError as e:
                if filter_errors:
                    logging.error(f"Patient filtered due to error calling 'label_names' for dataset '{self._name}', patient '{id}'.")
                    logging.error(f"Filtered error: {e}")
                    return False
                else:
                    raise e

            if ((isinstance(labels, str) and (labels == 'all' or labels in pat_labels)) or
                ((isinstance(labels, list) or isinstance(labels, np.ndarray) or isinstance(labels, tuple)) and len(np.intersect1d(labels, pat_labels)) != 0)):
                return True
            else:
                return False

        return fn

    def _hierarchical_exists(self) -> bool:
        """
        returns: True if the hierarchical dataset has been built.
        """
        # Check if folder exists.
        hier_path = os.path.join(self._path, 'hierarchical')
        return os.path.exists(hier_path)

    def _build_hierarchical(self) -> None:
        """
        effect: creates a hierarchical dataset based on dicom content, not existing structure.
        """
        logging.info('Building hierarchical dataset.')

        # Load all dicom files.
        raw_path = os.path.join(self._path, 'raw')
        dicom_files = []
        for root, _, files in os.walk(raw_path):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))

        # Copy dicom files.
        for f in sorted(dicom_files):
            # Get patient ID.
            dcm = dicom.read_file(f)
            pat_id = dcm.PatientID

            # Get modality.
            mod = dcm.Modality.lower()
            if not mod in ('ct', 'rtstruct'):
                continue

            # Create filepath.
            hier_path = os.path.join(self._path, 'hierarchical')
            filename = os.path.basename(f)
            filepath = os.path.join(hier_path, pat_id, mod, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save dicom.
            dcm.save_as(filepath)
