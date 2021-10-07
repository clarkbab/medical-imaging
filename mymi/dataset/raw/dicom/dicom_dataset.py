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
from typing import Callable, List, Optional, Union

from mymi import cache
from mymi import config
from mymi import logging
from mymi import regions
from mymi import types

from ...dataset import Dataset, DatasetType
from .dicom_patient import DICOMPatient
from .hierarchy import require_hierarchy
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
        self._path = os.path.join(config.directories.datasets, 'raw', name)

        # Load 'ct_from' flag.
        ct_from_name = None
        for f in os.listdir(self._path):
            match = re.match('^ct_from_(.*)$', f)
            if match:
                ct_from_name = match.group(1)

        self._ct_from = DICOMDataset(ct_from_name) if ct_from_name is not None else None
        self._global_id = f"DICOM: {name}"
        self._global_id = self._global_id + f" (CT from - {self._ct_from})" if self._ct_from is not None else self._global_id
        self._name = name
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self}' not found.")

        # Load region map.
        self._region_map = self._load_region_map()

    def cache_key(self) -> str:
        return self._global_id

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def ct_from(self) -> Optional['DICOMDataset']:
        return self._ct_from

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> DatasetType:
        return self._type

    @property
    def path(self) -> str:
        return self._path

    @require_hierarchy
    def trimmed_summary(self) -> pd.DataFrame:
        path = os.path.join(self._path, 'hierarchy', 'trimmed', 'summary.csv')
        return pd.read_csv(path)

    @require_hierarchy
    def has_patient(
        self,
        id: types.PatientID) -> bool:
        """
        returns: whether the patient is present in the dataset or not.
        args:
            id: the patient ID.
        """
        return id in self.list_patients()

    @require_hierarchy
    @cache.method('_global_id')
    def list_patients(
        self,
        regions: types.PatientRegions = 'all',
        trimmed: bool = False) -> List[str]:
        """
        returns: a list of patient IDs.
        """
        # Load top-level folders from 'hierarchy' dataset.
        if trimmed:
            path = os.path.join(self._path, 'hierarchy', 'trimmed', 'data')
        else:
            path = os.path.join(self._path, 'hierarchy', 'data')
        pats = list(sorted(os.listdir(path)))

        # Filter by 'regions'.
        pats = list(filter(self._filter_patient_by_regions(regions), pats))
        return pats

    @require_hierarchy
    def patient(
        self,
        id: types.PatientID,
        trimmed: bool = False) -> DICOMPatient:
        """
        returns: a DICOMPatient object.
        args:
            id: the patient ID.
        """
        if type(id) == int:
            id = str(id)
        ct_from = self._ct_from.patient(id, trimmed=trimmed) if self._ct_from is not None else None
        return DICOMPatient(self, id, ct_from=ct_from, region_map=self._region_map, trimmed=trimmed)

    @require_hierarchy
    @cache.method('_global_id')
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

    @require_hierarchy
    @cache.method('_global_id')
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

    @require_hierarchy
    @cache.method('_global_id')
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
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
            'orientation-row-x': float,
            'orientation-row-y': float,
            'orientation-row-z': float,
            'orientation-col-x': float,
            'orientation-col-y': float,
            'orientation-col-z': float,
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

    @require_hierarchy
    @cache.method('_global_id')
    def list_regions(
        self,
        clear_cache: bool = False,
        num_pats: Union[str, int] = 'all',
        pat_ids: types.PatientIDs = 'all',
        trimmed: bool = False,
        use_mapping: bool = True) -> pd.DataFrame:
        """
        returns: a DataFrame with patient region names.
        kwargs:
            clear_cache: force the cache to clear.
            num_pats: the number of patients to include.
            pat_ids: include listed patients.
            use_mapping: use region map if present.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pats = self.list_patients(trimmed=trimmed)

        # Filter patients.
        pats = list(filter(self._filter_patient_by_pat_ids(pat_ids), pats))
        pats = list(filter(self._filter_patient_by_num_pats(num_pats), pats))

        # Add patient regions.
        logging.info(f"Loading regions for dataset '{self._name}'..")
        for pat in tqdm(pats):
            try:
                pat_regions = self.patient(pat, trimmed=trimmed).list_regions(use_mapping=use_mapping)
            except ValueError as e:
                # Allow errors if we're inspecting 'trimmed' patients.
                if trimmed:
                    logging.error(e)
                else:
                    raise e

            for pat_region in pat_regions:
                data = {
                    'patient-id': pat,
                    'region': pat_region
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    @require_hierarchy
    @cache.method('_global_id')
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

    def _filter_patient_by_num_pats(
        self,
        num_pats: int) -> Callable[[str], bool]:
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

    def _filter_patient_by_pat_ids(
        self,
        pat_ids: Union[str, List[str]]) -> Callable[[str], bool]:
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
