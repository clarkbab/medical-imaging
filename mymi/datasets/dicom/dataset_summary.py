import numpy as np
import os
import pandas as pd
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientSummary

FLOAT_DP = 2

class DatasetSummary:
    def __init__(self, dataset=ds):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = ds

    def label_summary(self, num_pats='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing rows of label summaries.
        num_pats: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
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
            pat_sum = PatientSummary.from_id(pat_id, dataset=self.dataset)

            # Get RTSTRUCT details.
            rtstruct_details_df = pat_sum.rtstruct_details()

            # Add label counts.
            rtstruct_details_df['count'] = 1
            label_summary_df = label_summary_df.merge(rtstruct_details_df, how='outer', on='roi-label')
            label_summary_df['count'] = (label_summary_df['count_x'].fillna(0) + label_summary_df['count_y'].fillna(0)).astype(np.uint16)
            label_summary_df = label_summary_df.drop(['count_x', 'count_y'], axis=1)

        # Sort by 'roi-label'.
        label_summary_df = label_summary_df.sort_values('roi-label').reset_index(drop=True)

        return label_summary_df

    def patient_summary(self, num_pats='all', read_cache=True, write_cache=True):
        """
        returns: a dataframe containing rows of patient summaries.
        num_pats: the number of patients to summarise.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Define table structure.
        patient_summary_cols = {
            'dim-x': np.uint16,
            'dim-y': np.uint16,
            'dim-z': np.uint16,
            'fov-x': 'float64',
            'fov-y': 'float64',
            'fov-z': 'float64',
            'hu-min': 'float64',
            'hu-max': 'float64',
            'num-empty': np.uint16,
            'offset-x': 'float64',
            'offset-y': 'float64',
            'pat-id': 'object',
            'res-x': 'float64',
            'res-y': 'float64',
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
            pat_sum = PatientSummary.from_id(pat_id, dataset=self.dataset)

            # Get patient scan info.
            ct_details_df = pat_sum.ct_details()

            # Check for consistency among scans.
            assert len(ct_details_df['dim-x'].unique()) == 1
            assert len(ct_details_df['dim-y'].unique()) == 1
            assert len(ct_details_df['offset-x'].unique()) == 1
            assert len(ct_details_df['offset-y'].unique()) == 1
            assert len(ct_details_df['res-x'].unique()) == 1
            assert len(ct_details_df['res-y'].unique()) == 1
            assert len(ct_details_df['scale-int'].unique()) == 1
            assert len(ct_details_df['scale-slope'].unique()) == 1

            # Calculate res-z - this will be the smallest available diff.
            res_zs = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_details_df['offset-z'])])
            res_z = res_zs[0]

            # Calculate fov-z and dim-z.
            fov_z = ct_details_df['offset-z'].max() - ct_details_df['offset-z'].min()
            dim_z = int(fov_z / res_z) + 1

            # Calculate number of empty slices.
            num_slices = len(ct_details_df)
            num_empty = dim_z - num_slices

            # Get patient RTSTRUCT info.
            rtstruct_details_df = pat_sum.rtstruct_details()

            # Add table row.
            row_data = {
                'dim-x': ct_details_df['dim-x'][0],
                'dim-y': ct_details_df['dim-y'][0],
                'dim-z': dim_z,
                'fov-x': ct_details_df['dim-x'][0] * ct_details_df['res-x'][0],
                'fov-y': ct_details_df['dim-y'][0] * ct_details_df['res-y'][0],
                'fov-z': dim_z * res_z,
                'hu-min': ct_details_df['hu-min'].min(),
                'hu-max': ct_details_df['hu-max'].max(),
                'num-empty': num_empty,
                'offset-x': ct_details_df['offset-x'][0],
                'offset-y': ct_details_df['offset-y'][0],
                'pat-id': pat_id,
                'res-x': ct_details_df['res-x'][0],
                'res-y': ct_details_df['res-y'][0],
                'res-z': res_z, 
                'roi-num': len(rtstruct_details_df),
                'scale-int': ct_details_df['scale-int'][0],
                'scale-slope': ct_details_df['scale-slope'][0],
            }
            patient_summary_df = patient_summary_df.append(row_data, ignore_index=True)

        # Set index.
        patient_summary_df = patient_summary_df.set_index('pat-id')

        return patient_summary_df

        