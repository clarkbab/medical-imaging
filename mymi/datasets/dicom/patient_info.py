import os
import numpy as np
import pandas as pd
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.cache import DataCache

CACHE_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'cache')
FLOAT_DP = 2

class PatientInfo:
    def __init__(self, pat_id, dataset=ds, verbose=False):
        """
        pat_id: a patient ID string.
        dataset: a DICOM dataset.
        """
        self.cache = DataCache(CACHE_ROOT)
        self.dataset = dataset
        self.pat_id = pat_id
        # TODO: Add logger class.
        self.verbose = verbose

    def ct_info(self, read_cache=True, write_cache=True):
        """
        returns: dataframe with rows containing CT info.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'patient_info',
            'method': 'ct_info',
            'patient_id': self.pat_id
        }
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache key {key}.")
                return self.cache.read(key, 'dataframe')
            
        # Define dataframe structure.
        detail_cols = {
            'hu-min': 'float64',
            'hu-max': 'float64',
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'res-x': np.uint16,
            'res-y': np.uint16,
            'scale-int': 'float64',
            'scale-slope': 'float64',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
        }
        info_df = pd.DataFrame(columns=detail_cols.keys())

        ct_dicoms = self.dataset.list_ct(self.pat_id)
        
        # Add info.
        for ct_dicom in ct_dicoms:
            # Perform scaling from stored values to HU.
            hus = ct_dicom.pixel_array * ct_dicom.RescaleSlope + ct_dicom.RescaleIntercept

            row_data = {
               'hu-min': hus.min(),
               'hu-max': hus.max(),
               'offset-x': ct_dicom.ImagePositionPatient[0], 
               'offset-y': ct_dicom.ImagePositionPatient[1], 
               'offset-z': ct_dicom.ImagePositionPatient[2], 
               'res-x': ct_dicom.pixel_array.shape[1],  # Pixel array is stored (y, x) for plotting.
               'res-y': ct_dicom.pixel_array.shape[0],
               'scale-int': ct_dicom.RescaleIntercept,
               'scale-slope': ct_dicom.RescaleSlope,
               'spacing-x': ct_dicom.PixelSpacing[0],
               'spacing-y': ct_dicom.PixelSpacing[1]
            }
            info_df = info_df.append(row_data, ignore_index=True)

        # Set column types as 'append' crushes them.
        info_df = info_df.astype(detail_cols)

        # Sort by 'offset-z'.
        info_df = info_df.sort_values('offset-z').reset_index(drop=True)

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, info_df, 'dataframe')

        return info_df

    def full_info(self, read_cache=True, write_cache=True):
        """
        returns: dataframe with single row summary of CT images.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'patient_info',
            'method': 'full_info',
            'patient_id': self.pat_id
        }
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache key {key}.")
                return self.cache.read(key, 'dataframe')

        # Define table structure.
        full_info_cols = {
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
        full_info_df = pd.DataFrame(columns=full_info_cols.keys())

        # Get patient scan info.
        ct_info_df = self.ct_info(read_cache=read_cache, write_cache=write_cache)

        # Check for consistency among scans.
        assert len(ct_info_df['res-x'].unique()) == 1
        assert len(ct_info_df['res-y'].unique()) == 1
        assert len(ct_info_df['offset-x'].unique()) == 1
        assert len(ct_info_df['offset-y'].unique()) == 1
        assert len(ct_info_df['spacing-x'].unique()) == 1
        assert len(ct_info_df['spacing-y'].unique()) == 1
        assert len(ct_info_df['scale-int'].unique()) == 1
        assert len(ct_info_df['scale-slope'].unique()) == 1

        # Calculate spacing-z - this will be the smallest available diff.
        spacings_z = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_info_df['offset-z'])])
        spacing_z = spacings_z[0]

        # Calculate fov-z and res-z.
        fov_z = ct_info_df['offset-z'].max() - ct_info_df['offset-z'].min()
        res_z = int(round(fov_z / spacing_z, 0) + 1)

        # Calculate number of empty slices.
        num_slices = len(ct_info_df)
        num_empty = res_z - num_slices

        # Get patient RTSTRUCT info.
        rtstruct_info_df = self.rtstruct_info(read_cache=read_cache, write_cache=write_cache)

        # Add table row.
        row_data = {
            'res-x': ct_info_df['res-x'][0],
            'res-y': ct_info_df['res-y'][0],
            'res-z': res_z,
            'fov-x': ct_info_df['res-x'][0] * ct_info_df['spacing-x'][0],
            'fov-y': ct_info_df['res-y'][0] * ct_info_df['spacing-y'][0],
            'fov-z': res_z * spacing_z,
            'hu-min': ct_info_df['hu-min'].min(),
            'hu-max': ct_info_df['hu-max'].max(),
            'num-empty': num_empty,
            'offset-x': ct_info_df['offset-x'][0],
            'offset-y': ct_info_df['offset-y'][0],
            'offset-z': ct_info_df['offset-z'][0],
            'pat-id': self.pat_id,
            'spacing-x': ct_info_df['spacing-x'][0],
            'spacing-y': ct_info_df['spacing-y'][0],
            'spacing-z': spacing_z, 
            'roi-num': len(rtstruct_info_df),
            'scale-int': ct_info_df['scale-int'][0],
            'scale-slope': ct_info_df['scale-slope'][0],
        }
        full_info_df = full_info_df.append(row_data, ignore_index=True)

        # Set index.
        full_info_df = full_info_df.set_index('pat-id')

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, full_info_df, 'dataframe')

        return full_info_df

    def rtstruct_info(self, read_cache=True, write_cache=True):
        """
        returns: dataframe with row for each region-of-interest.
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        """
        # Load from cache if present.
        key = {
            'class': 'patient_info',
            'method': 'rtstruct_info',
            'patient_id': self.pat_id
        }
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache key {key}.")
                return self.cache.read(key, 'dataframe')
        
        # Define table structure.
        info_cols = {
            'roi-label': 'object'
        }
        info_df = pd.DataFrame(columns=info_cols.keys())

        rois = self.dataset.get_rtstruct(self.pat_id).StructureSetROISequence
        
        # Add info for each region-of-interest.
        for roi in rois:
            row_data = {
                'roi-label': roi.ROIName
            }
            info_df = info_df.append(row_data, ignore_index=True)

        # Set column type.
        info_df = info_df.astype(info_cols)

        # Sort by label.
        info_df = info_df.sort_values('roi-label').reset_index(drop=True)

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, info_df, 'dataframe')

        return info_df
