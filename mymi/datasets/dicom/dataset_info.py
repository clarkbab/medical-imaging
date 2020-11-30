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

    def label_info(self, num_pats='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing rows of label summaries.
        num_pats: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset_info',
            'method': 'label_info'
        }
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache key {key}.")
                return self.cache.read(key, 'dataframe')

        # Define table structure.
        label_info_cols = {
            'count': np.uint16,
            'roi-label': 'object'
        }
        label_info_df = pd.DataFrame(columns=label_info_cols.keys())

        # List patients.
        pat_ids = self.dataset.list_patients()

        # Run on subset of patients.
        if num_pats != 'all':
            pat_ids = pat_ids[:num_pats]

        for pat_id in tqdm(pat_ids):
            # Get patient info.
            pat_info = PatientInfo(pat_id, dataset=self.dataset, verbose=self.verbose)

            # Get RTSTRUCT info.
            rtstruct_info_df = pat_info.rtstruct_info(read_cache=read_cache, write_cache=write_cache)

            # Add label counts.
            rtstruct_info_df['count'] = 1
            label_info_df = label_info_df.merge(rtstruct_info_df, how='outer', on='roi-label')
            label_info_df['count'] = (label_info_df['count_x'].fillna(0) + label_info_df['count_y'].fillna(0)).astype(np.uint16)
            label_info_df = label_info_df.drop(['count_x', 'count_y'], axis=1)

        # Sort by 'roi-label'.
        label_info_df = label_info_df.sort_values('roi-label').reset_index(drop=True)

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, label_info_df, 'dataframe')

        return label_info_df

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
                if self.verbose: print(f"Reading cache key {key}.")
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
            'num-empty': np.uint16,
            'offset-x': 'float64',
            'offset-y': 'float64',
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
            info_df = pat_info.full_info()
            patient_info_df = patient_info_df.append(info_df)

        # Set index.
        patient_info_df = patient_info_df.drop('pat-id', axis=1)
        patient_info_df.index = patient_info_df.index.rename('pat-id')

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, patient_info_df, 'dataframe')

        return patient_info_df

        