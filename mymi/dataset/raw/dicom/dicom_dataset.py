from collections import OrderedDict
import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydicom as dicom
import re
from scipy.ndimage import center_of_mass
import shutil
from skimage.draw import polygon
from torchio import ScalarImage, Subject
from tqdm import tqdm
from typing import Callable, Optional, Sequence, Union

from mymi import cache
from mymi import config
from mymi import logging
from mymi import regions
from mymi import types

from ...dataset import Dataset, DatasetType
from .dicom_patient import DICOMPatient
from .region_map import RegionMap

Z_SPACING_ROUND_DP = 2

class DICOMDataset(Dataset):
    def __init__(
        self,
        name: str):
        """
        args:
            name: the name of the dataset.
        """
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'raw', name)

        # Load 'ct_from' flag.
        self._ct_from = None
        for f in os.listdir(self._path):
            match = re.match('^ct_from_(.*)$', f)
            if match:
                self._ct_from = match.group(1)

        # Check if datasets exist.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{name}' not found.")
        if self._ct_from:
            ct_path = os.path.join(config.directories.datasets, 'raw', self._ct_from)
            if not os.path.exists(ct_path):
                raise ValueError(f"Dataset '{self._ct_from}' not found.")

        # Load region map.
        self._region_map = self._load_region_map()

    @property
    def description(self) -> str:
        return f"DICOM: {self._name}"

    @property
    def ct_from(self) -> Optional[str]:
        return self._ct_from

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> DatasetType:
        return DatasetType.DICOM

    @property
    def path(self) -> str:
        return self._path

    def _require_hierarchical(fn: Callable) -> Callable:
        """
        effect: returns a wrapped function, ensuring hierarchical data has been built.
        args:
            fn: the wrapped function.
        """
        def _require_hierarchical_wrapper(self, *args, **kwargs):
            if not self._hierarchical_exists():
                self._build_hierarchical()
                self._trim_hierarchical()
            return fn(self, *args, **kwargs)
        return _require_hierarchical_wrapper

    @_require_hierarchical
    def has_patient(
        self,
        id: types.PatientID) -> bool:
        """
        returns: whether the patient is present in the dataset or not.
        args:
            id: the patient ID.
        """
        return id in self.list_patients()

    @_require_hierarchical
    def list_patients(
        self,
        regions: types.PatientRegions = 'all') -> Sequence[str]:
        """
        returns: a list of patient IDs.
        """
        # Load top-level folders from 'hierarchical' dataset.
        hier_path = os.path.join(self._path, 'hierarchical', 'data')
        pats = list(sorted(os.listdir(hier_path)))

        # Filter by 'regions'.
        pats = list(filter(self._filter_patient_by_regions(regions), pats))
        return pats

    @_require_hierarchical
    def patient(
        self,
        id: types.PatientID) -> DICOMPatient:
        """
        returns: a DICOMPatient object.
        args:
            id: the patient ID.
        """
        # Convert to string.
        if type(id) == int:
            id = str(id)

        # Check that patient ID exists.
        pat_path = os.path.join(self._path, 'hierarchical', 'data', id)
        if not os.path.isdir(pat_path):
            raise ValueError(f"Patient '{id}' not found in dataset '{self._name}'.")

        # Create patient.
        pat = DICOMPatient(self._name, id, ct_from=self._ct_from, region_map=self._region_map)
        
        return pat

    @_require_hierarchical
    @cache.method('_ct_from', '_name')
    def info(
        self, 
        clear_cache: bool = False,
        num_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame with patient info.
        kwargs:
            clear_cache: force the cache to clear.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
            regions: include patients with (at least) on of the regions.
            use_mapping: use region map if present.
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
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        logging.info(f"Filtering on region names for dataset '{self._name}'..")
        pats = list(filter(self._filter_patient_by_regions(regions, use_mapping=use_mapping), tqdm(pats)))
        pats = list(filter(self._filter_patient_by_num_pats(num_pats), pats))

        # Add patient regions.
        for pat in tqdm(pats):
            info_df = self.patient(pat).info(clear_cache=clear_cache)
            info_df['patient-id'] = pat
            df = df.append(info_df)

        # Set column type.
        df = df.astype(cols)

        return df

    @_require_hierarchical
    @cache.method('_ct_from', '_name')
    def ct_distribution(
        self, 
        bin_width: int = 10,
        clear_cache: bool = False,
        num_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        regions: types.PatientRegions = 'all') -> OrderedDict:
        """
        effect: plots CT distribution of the dataset.
        kwargs:
            bin_width: the width of the histogram bins.
            clear_cache: forces the cache to clear.
            num_pats: the number of patients to include.
            pat_ids: the patients to include.
            regions: include patients with any of the listed regions (behaves like an OR).
        """
        # Load all patients.
        pats = self.list_patients()
        
        # Filter patients.
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        logging.info(f"Filtering on region names for dataset '{self._name}'..")
        pats = list(filter(self._filter_patient_by_regions(regions), tqdm(pats)))
        pats = list(filter(self._filter_patient_by_num_pats(num_pats), pats))

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
    @cache.method('_ct_from', '_name')
    def ct_summary(
        self, 
        clear_cache: bool = False,
        num_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame with patient CT summaries.
        kwargs:
            clear_cache: force the cache to clear.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
            regions: include patients with (at least) on of the regions.
            use_mapping: use region map if present.
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
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        logging.info(f"Filtering on region names for dataset '{self._name}'..")
        pats = list(filter(self._filter_patient_by_regions(regions, use_mapping=use_mapping), tqdm(pats)))
        pats = list(filter(self._filter_patient_by_num_pats(num_pats), pats))

        # Add patient info.
        logging.info(f"Loading CT summary for dataset '{self._name}'..")
        for pat in tqdm(pats):
            # Load patient CT info.
            pat_df = self.patient(pat).ct_summary(clear_cache=clear_cache)

            # Add row.
            pat_df['patient-id'] = pat
            df = df.append(pat_df)

        # Set column type.
        df = df.astype(cols)

        return df

    @_require_hierarchical
    @cache.method('_name')
    def list_regions(
        self,
        num_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame with patient region names.
        kwargs:
            clear_cache: force the cache to clear.
            num_pats: the number of patients to include.
            pat_ids: include listed patients.
            regions: include patients with (at least) on of the regions.
            use_mapping: use region map if present.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        logging.info(f"Filtering on region names for dataset '{self._name}'..")
        pats = list(filter(self._filter_patient_by_regions(regions, use_mapping=use_mapping), tqdm(pats)))
        pats = list(filter(self._filter_patient_by_num_pats(num_pats), pats))

        # Add patient regions.
        logging.info(f"Loading region names for dataset '{self._name}'..")
        for pat in tqdm(pats):
            for pat_region in self.patient(pat).list_regions(use_mapping=use_mapping):
                data = {
                    'patient-id': pat,
                    'region': pat_region
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    @_require_hierarchical
    @cache.method('_ct_from', '_name')
    def region_summary(
        self, 
        clear_cache: bool = False,
        num_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        regions: types.PatientRegions = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame with patient regions and information.
        kwargs:
            clear_cache: force the cache to clear.
            regions: include patients with (at least) on of the requested regions.
            num_pats: the number of patients to summarise.
            pat_ids: include listed patients.
            use_mapping: use region map if present.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
            'centroid-mm-x': float,
            'centroid-mm-y': float,
            'centroid-mm-z': float,
            'centroid-voxels-x': float,
            'centroid-voxels-y': float,
            'centroid-voxels-z': float,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
            'width-voxels-x': float,
            'width-voxels-y': float,
            'width-voxels-z': float,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients()

        # Filter patients.
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        logging.info(f"Filtering on region names for dataset '{self._name}'..")
        pats = list(filter(self._filter_patient_by_regions(regions, use_mapping=use_mapping), pats))
        pats = list(filter(self._filter_patient_by_num_pats(num_pats), pats))

        # Add patient regions.
        logging.info(f"Loading region summary for dataset '{self._name}'..")
        for pat in tqdm(pats):
            # Load patient summary.
            summary_df = self.patient(pat).region_summary(clear_cache=clear_cache, regions=regions, use_mapping=use_mapping)

            # Add rows.
            for _, row in summary_df.iterrows():
                data = {
                    'patient-id': pat,
                    'region': row.region,
                    'width-mm-x': row['width-mm-x'],
                    'width-mm-y': row['width-mm-y'],
                    'width-mm-z': row['width-mm-z'],
                    'centroid-mm-x': row['centroid-mm-x'],
                    'centroid-mm-y': row['centroid-mm-y'],
                    'centroid-mm-z': row['centroid-mm-z'],
                    'centroid-voxels-x': row['centroid-voxels-x'],
                    'centroid-voxels-y': row['centroid-voxels-y'],
                    'centroid-voxels-z': row['centroid-voxels-z'],
                    'width-mm-x': row['width-mm-x'],
                    'width-mm-y': row['width-mm-y'],
                    'width-mm-z': row['width-mm-z'],
                    'width-voxels-x': row['width-voxels-x'],
                    'width-voxels-y': row['width-voxels-y'],
                    'width-voxels-z': float,
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    def _load_region_map(self) -> Optional[RegionMap]:
        """
        returns: a RegionMap object mapping dataset region names to internal names.
        raises:
            ValueError: if 'region-map.csv' isn't configured properly.
        """
        # Check for region map.
        filepath = os.path.join(self._path, 'region-map.csv')
        if os.path.exists(filepath):
            # Load map file.
            map_df = pd.read_csv(filepath)

            # Check that internal region names are entered correctly.
            for n in map_df.internal:
                if not regions.is_region(n):
                    raise ValueError(f"Error in region map for dataset '{self._name}', '{n}' is not an internal region.")
            
            return RegionMap(map_df)
        else:
            return None

    @classmethod
    def ct_statistics(cls, regions='all'):
        """
        returns: a dataframe of CT statistics for the entire dataset.
        kwargs:
            regions: only include data for patients with the regions.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'kwargs': {
                'regions': regions
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result

        # Convert 'regions'.
        if isinstance(regions, str) and regions != 'all':
            regions = [regions]

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
            # Get patient regions.
            pat_regions = list(cls.patient_regions(pat_id)['region'])
            
            # Skip if patient has no regions, or doesn't have the specified regions.
            if len(pat_regions) == 0 or (regions != 'all' and not np.array_equal(np.intersect1d(regions, pat_regions), regions)):
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
        # Load all dicom files.
        raw_path = os.path.join(self._path, 'raw')
        if not os.path.exists(raw_path):
            raise ValueError(f"No 'raw' folder found for dataset '{self.description}'.")
        dicom_files = []
        for root, _, files in os.walk(raw_path):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))

        # Copy dicom files.
        logging.info(f"Building hierarchical dataset for '{self._name}'..")
        for f in tqdm(sorted(dicom_files)):
            # Get patient ID.
            dcm = dicom.read_file(f)
            pat_id = dcm.PatientID

            # Get modality.
            mod = dcm.Modality.lower()
            if not mod in ('ct', 'rtstruct'):
                continue

            # Get series UID.
            series_UID = dcm.SeriesInstanceUID

            # Create filepath.
            filename = os.path.basename(f)
            filepath = os.path.join(self._path, 'hierarchical', 'data', pat_id, mod, series_UID, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save dicom.
            dcm.save_as(filepath)

    def _trim_hierarchical(self) -> None:
        """
        effect: removes patients that don't have RTSTRUCT/CT DICOMS.
        """
        logging.info(f"Removing invalid patients from hierarchical dataset for '{self._name}'..")

        # Get patients.
        pats = self.list_patients()

        # Trim each patient.
        for pat in tqdm(pats):
            try:
                # Creating the patient raises errors for missing files.
                patient = self.patient(pat)

                # Loading CT summary ensures data is consistent.
                patient.ct_summary(clear_cache=True)

            except ValueError as e:
                # Move patient to error folder.
                pat_path = os.path.join(self._path, 'hierarchical', 'data', pat)
                pat_error_path = os.path.join(self._path, 'hierarchical', 'errors', pat)
                shutil.move(pat_path, pat_error_path)

                # Write error message.
                msg = f"Patient '{pat}' removed from hierarchical dataset due to error."
                error_msg = f"Error: {e}"
                filepath = os.path.join(pat_error_path, 'error.log')
                with open(filepath, 'w') as f:
                    f.write(msg + '\n')
                    f.write(error_msg + '\n')

                # Log error message.
                logging.error(msg)
                logging.error(error_msg)

    def _filter_patient_by_num_pats(num_pats: int) -> Callable[[str], bool]:
        """
        returns: a function to filter patients by number of patients allowed.
        args:
            num_pats: the number of patients to keep.
        """
        def fn(id):
            if num_pats == 'all' or fn.num_included < num_pats:
                fn.num_included += 1
                return True
            else:
                return False

        # Assign state to the function.
        fn.num_included = 0
        return fn

    def _filter_patient_by_pat_ids(pat_ids: Union[str, Sequence[str]]) -> Callable[[str], bool]:
        """
        returns: a function to filter patients based on a 'pat_ids' string or list/tuple.
        args:
            pat_ids: the passed 'pat_ids' kwarg.
        """
        def fn(id):
            if ((isinstance(pat_ids, str) and (pat_ids == 'all' or id == pat_ids)) or
                ((isinstance(pat_ids, list) or isinstance(pat_ids, np.ndarray) or isinstance(pat_ids, tuple)) and id in pat_ids)):
                return True
            else:
                return False
        return fn

    def _filter_patient_by_regions(
        self,
        regions: types.PatientRegions,
        use_mapping: bool = True) -> Callable[[str], bool]:
        """
        returns: a function that filters patients on region presence.
        args:
            regions: the passed 'regions' kwarg.
        kwargs:
            clear_cache: force the cache to clear.
            use_mapping: use region map if present.
        """
        def fn(id):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.patient(id).has_region(regions, use_mapping=use_mapping)
            else:
                pat_regions = self.patient(id).list_regions(use_mapping=use_mapping)
                if len(np.intersect1d(regions, pat_regions)) != 0:
                    return True
                else:
                    return False
        return fn
