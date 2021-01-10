import os
import numpy as np
import pandas as pd
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import cache
from mymi import dataset

FLOAT_DP = 2

class PatientInfo:
    def __init__(self, pat_id):
        """
        pat_id: a patient ID string.
        """
        self.pat_id = pat_id

    def ct_info(self):
        """
        returns: dataframe with rows containing CT info.
        """
        # Load from cache if present.
        key = {
            'class': 'patient_info',
            'method': 'ct_info',
            'patient_id': self.pat_id
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
            
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

        ct_dicoms = dataset.list_ct(self.pat_id)
        
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

        if cache.write_enabled():
            cache.write(key, info_df, 'dataframe')

        return info_df

    def full_info(self):
        """
        returns: dataframe with single row summary of CT images.
        """
        # Load from cache if present.
        key = {
            'class': 'patient_info',
            'method': 'full_info',
            'patient_id': self.pat_id
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')

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
        ct_info_df = self.ct_info()

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
        num_missing = res_z - num_slices

        # Get patient RTSTRUCT info.
        region_info_df = self.region_info()

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
            'num-missing': num_missing,
            'offset-x': ct_info_df['offset-x'][0],
            'offset-y': ct_info_df['offset-y'][0],
            'offset-z': ct_info_df['offset-z'][0],
            'pat-id': self.pat_id,
            'spacing-x': ct_info_df['spacing-x'][0],
            'spacing-y': ct_info_df['spacing-y'][0],
            'spacing-z': spacing_z, 
            'roi-num': len(region_info_df),
            'scale-int': ct_info_df['scale-int'][0],
            'scale-slope': ct_info_df['scale-slope'][0],
        }
        full_info_df = full_info_df.append(row_data, ignore_index=True)

        # Set index.
        full_info_df = full_info_df.set_index('pat-id')

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, full_info_df, 'dataframe')

        return full_info_df

    def region_info(self):
        """
        returns: dataframe with row for each region.
        """
        # Load from cache if present.
        key = {
            'class': 'patient_info',
            'method': 'region_info',
            'patient_id': self.pat_id
        }
        if cache.read_enabled() and cache.exists(key):
            return cache.read(key, 'dataframe')
        
        # Define table structure.
        region_info_cols = {
            'region': 'object'
        }
        region_info_df = pd.DataFrame(columns=region_info_cols.keys())

        rois = dataset.get_rtstruct(self.pat_id).StructureSetROISequence
        
        # Add info for each region-of-interest.
        for roi in rois:
            row_data = {
                'region': roi.ROIName
            }
            region_info_df = region_info_df.append(row_data, ignore_index=True)

        # Set column type.
        region_info_df = region_info_df.astype(region_info_cols)

        # Sort by label.
        region_info_df = region_info_df.sort_values('region').reset_index(drop=True)

        # Write data to cache.
        if cache.write_enabled():
            cache.write(key, region_info_df, 'dataframe')

        return region_info_df
