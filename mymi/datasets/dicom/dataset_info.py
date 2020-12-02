import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.cache import DataCache
from mymi.datasets.dicom import PatientInfo

CACHE_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'cache')
FLOAT_DP = 2

class DatasetInfo:
    def __init__(self, dataset=ds, verbose=False):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = ds
        self.cache = DataCache(CACHE_ROOT)
        self.verbose = verbose

    def region_count(self, num_patients='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing regions and num patients with region.
        num_patients: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset_info',
            'method': 'region_count'
        }
        if read_cache:
            if self.cache.exists(key):
                return self.cache.read(key, 'dataframe')

        # Define table structure.
        region_count_cols = {
            'num-patients': np.uint16,
            'region': 'object'
        }
        region_count_df = pd.DataFrame(columns=region_count_cols.keys())

        # List patients.
        pat_ids = self.dataset.list_patients()

        # Run on subset of patients.
        if num_patients != 'all':
            pat_ids = pat_ids[:num_patients]

        for pat_id in tqdm(pat_ids):
            # Get patient info.
            pat_info = PatientInfo(pat_id, dataset=self.dataset, verbose=self.verbose)

            # Get RTSTRUCT info.
            region_info_df = pat_info.region_info(read_cache=read_cache, write_cache=write_cache)

            # Add label counts.
            region_info_df['num-patients'] = 1
            region_count_df = region_count_df.merge(region_info_df, how='outer', on='region')
            region_count_df['num-patients'] = (region_count_df['num-patients_x'].fillna(0) + region_count_df['num-patients_y'].fillna(0)).astype(np.uint16)
            region_count_df = region_count_df.drop(['num-patients_x', 'num-patients_y'], axis=1)

        # Sort by 'roi-label'.
        region_count_df = region_count_df.sort_values('region').reset_index(drop=True)

        # Write data to cache.
        if write_cache:
            self.cache.write(key, region_count_df, 'dataframe')

        return region_count_df

    def patient_info(self, num_pats='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing rows of patient summaries.
        num_pats: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset_info',
            'method': 'patient_info'
        }
        if read_cache:
            if self.cache.exists(key):
                return self.cache.read(key, 'dataframe')
                
        # Define table structure.
        patient_info_cols = {
            'res-x': np.uint16,
            'res-y': np.uint16,
            'res-z': np.uint16,
            'fov-x': 'float64',
            'fov-y': 'float64',
            'fov-z': 'float64',
            'hu-min': 'float64',
            'hu-max': 'float64',
            'num-missing': np.uint16,
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'pat-id': 'object',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
            'spacing-z': 'float64',
            'roi-num': np.uint16,
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        patient_info_df = pd.DataFrame(columns=patient_info_cols.keys())

        # List patients.
        pat_ids = self.dataset.list_patients()

        # Run on subset of patients.
        if num_pats != 'all':
            pat_ids = pat_ids[:num_pats]

        for pat_id in tqdm(pat_ids):
            # Get patient info.
            pat_info = PatientInfo(pat_id, dataset=ds, verbose=self.verbose)
            info_df = pat_info.full_info(read_cache=read_cache, write_cache=write_cache)
            patient_info_df = patient_info_df.append(info_df)

        # Set index.
        patient_info_df = patient_info_df.drop('pat-id', axis=1)
        patient_info_df.index = patient_info_df.index.rename('pat-id')

        # Write data to cache.
        if write_cache:
            self.cache.write(key, patient_info_df, 'dataframe')

        return patient_info_df

    def patient_regions(self, read_cache=True, write_cache=True):
        """
        returns: a dataframe linking patients to contoured regions.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset_info',
            'method': 'patient_regions'
        }
        if read_cache:
            if self.cache.exists(key):
                return self.cache.read(key, 'dataframe')
                
        # Define table structure.
        patient_regions_cols = {
            'patient-id': 'object',
            'region': 'object'
        }
        patient_regions_df = pd.DataFrame(columns=patient_regions_cols.keys())

        # Load each patient.
        pat_ids = self.dataset.list_patients()

        for pat_id in tqdm(pat_ids):
            # Get rtstruct info.
            pat_info = PatientInfo(pat_id, dataset=ds, verbose=self.verbose)
            region_info_df = pat_info.region_info(read_cache=read_cache, write_cache=write_cache)

            # Add rows.
            for _, row in region_info_df.iterrows():
                row_data = {
                    'patient-id': pat_id,
                    'region': row['region']
                }
                patient_regions_df = patient_regions_df.append(row_data, ignore_index=True)

        # Write data to cache.
        if write_cache:
            self.cache.write(key, patient_regions_df, 'dataframe')

        return patient_regions_df
