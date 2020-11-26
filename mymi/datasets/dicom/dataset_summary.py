import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.cache import DataCache
from mymi.datasets.dicom import PatientSummary

CACHE_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'cache')
FLOAT_DP = 2

class DatasetSummary:
    def __init__(self, dataset=ds, verbose=False):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = ds
        self.cache = DataCache(CACHE_ROOT)
        self.verbose = verbose

    def label_summary(self, num_pats='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing rows of label summaries.
        num_pats: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = f"dataset_summary:label_summary"
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache key '{key}'.")
                return self.cache.read(key, 'dataframe')

        # Define table structure.
        label_summary_cols = {
            'count': np.uint16,
            'roi-label': 'object'
        }
        label_summary_df = pd.DataFrame(columns=label_summary_cols.keys())

        # List patients.
        pat_ids = self.dataset.list_patients()

        # Run on subset of patients.
        if num_pats != 'all':
            pat_ids = pat_ids[:num_pats]

        for pat_id in tqdm(pat_ids):
            # Create patient summary.
            pat_sum = PatientSummary(pat_id, dataset=self.dataset, verbose=self.verbose)

            # Get RTSTRUCT details.
            rtstruct_details_df = pat_sum.rtstruct_details(read_cache=read_cache, write_cache=write_cache)

            # Add label counts.
            rtstruct_details_df['count'] = 1
            label_summary_df = label_summary_df.merge(rtstruct_details_df, how='outer', on='roi-label')
            label_summary_df['count'] = (label_summary_df['count_x'].fillna(0) + label_summary_df['count_y'].fillna(0)).astype(np.uint16)
            label_summary_df = label_summary_df.drop(['count_x', 'count_y'], axis=1)

        # Sort by 'roi-label'.
        label_summary_df = label_summary_df.sort_values('roi-label').reset_index(drop=True)

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key '{key}'.")
            self.cache.write(key, label_summary_df, 'dataframe')

        return label_summary_df

    def patient_summary(self, num_pats='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing rows of patient summaries.
        num_pats: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'dataset_summary',
            'method': 'patient_summary'
        }
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache key {key}.")
                return self.cache.read(key, 'dataframe')
                
        # Define table structure.
        patient_summary_cols = {
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
            'res-z': 'float64',
            'roi-num': np.uint16,
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        patient_summary_df = pd.DataFrame(columns=patient_summary_cols.keys())

        # List patients.
        pat_ids = self.dataset.list_patients()

        # Run on subset of patients.
        if num_pats != 'all':
            pat_ids = pat_ids[:num_pats]

        for pat_id in tqdm(pat_ids):
            # Create patient summary.
            pat_sum = PatientSummary(pat_id, dataset=ds, verbose=self.verbose)

            # Get patient scan info.
            ct_details_df = pat_sum.ct_details(read_cache=read_cache, write_cache=write_cache)

            # Check for consistency among scans.
            assert len(ct_details_df['res-x'].unique()) == 1
            assert len(ct_details_df['res-y'].unique()) == 1
            assert len(ct_details_df['offset-x'].unique()) == 1
            assert len(ct_details_df['offset-y'].unique()) == 1
            assert len(ct_details_df['spacing-x'].unique()) == 1
            assert len(ct_details_df['spacing-y'].unique()) == 1
            assert len(ct_details_df['scale-int'].unique()) == 1
            assert len(ct_details_df['scale-slope'].unique()) == 1

            # Calculate spacing-z - this will be the smallest available diff.
            spacings_z = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_details_df['offset-z'])])
            spacing_z = spacings_z[0]

            # Calculate fov-z and res-z.
            fov_z = ct_details_df['offset-z'].max() - ct_details_df['offset-z'].min()
            res_z = int(round(fov_z / spacing_z, 0) + 1)

            # Calculate number of empty slices.
            num_slices = len(ct_details_df)
            num_empty = res_z - num_slices

            # Get patient RTSTRUCT info.
            rtstruct_details_df = pat_sum.rtstruct_details(read_cache=read_cache, write_cache=write_cache)

            # Add table row.
            row_data = {
                'res-x': ct_details_df['res-x'][0],
                'res-y': ct_details_df['res-y'][0],
                'res-z': res_z,
                'fov-x': ct_details_df['res-x'][0] * ct_details_df['spacing-x'][0],
                'fov-y': ct_details_df['res-y'][0] * ct_details_df['spacing-y'][0],
                'fov-z': res_z * spacing_z,
                'hu-min': ct_details_df['hu-min'].min(),
                'hu-max': ct_details_df['hu-max'].max(),
                'num-empty': num_empty,
                'offset-x': ct_details_df['offset-x'][0],
                'offset-y': ct_details_df['offset-y'][0],
                'pat-id': pat_id,
                'spacing-x': ct_details_df['spacing-x'][0],
                'spacing-y': ct_details_df['spacing-y'][0],
                'spacing-z': spacing_z, 
                'roi-num': len(rtstruct_details_df),
                'scale-int': ct_details_df['scale-int'][0],
                'scale-slope': ct_details_df['scale-slope'][0],
            }
            patient_summary_df = patient_summary_df.append(row_data, ignore_index=True)

        # Set index.
        patient_summary_df = patient_summary_df.set_index('pat-id')

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, patient_summary_df, 'dataframe')

        return patient_summary_df

        