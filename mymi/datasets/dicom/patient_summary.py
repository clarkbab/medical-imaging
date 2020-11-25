import os
import numpy as np
import pandas as pd
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds

class PatientSummary:
    @staticmethod
    def from_id(pat_id, dataset=ds):
        """
        pat_id: an identifier for the patient.
        returns: PatientSummary object.
        """
        if not dataset.has_id(pat_id):
            print(f"Patient ID '{pat_id}' not found in dataset.")
            # raise error
            exit(0)

        # TODO: Read an env var or something to make dataset implicit.
        
        return PatientSummary(pat_id, dataset=dataset)

    def __init__(self, pat_id, dataset=ds):
        """
        pat_id: a patient ID string.
        dataset: a DICOM dataset.
        """
        self.dataset = dataset
        self.pat_id = pat_id

    def ct_details(self, read_cache=True, write_cache=True):
        """
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        returns: dataframe with info for each CT slice. 
        """
        # TODO: Add caching.
        # if read_cache:
            # Check if cache exists using SummaryCache object.
            # If present, return as is, else progress with calculation.
            # return details_df
            
        # Define dataframe structure.
        detail_cols = {
            'dim-x': np.uint16,
            'dim-y': np.uint16,
            'hu-min': 'float64',
            'hu-max': 'float64',
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'res-x': 'float64',
            'res-y': 'float64',
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        details_df = pd.DataFrame(columns=detail_cols.keys())

        ct_dicoms = self.dataset.list_ct(self.pat_id)
        
        # Add details.
        for ct_dicom in ct_dicoms:
            # Perform scaling from stored values to HU.
            hus = ct_dicom.pixel_array * ct_dicom.RescaleSlope + ct_dicom.RescaleIntercept

            row_data = {
               'dim-x': ct_dicom.pixel_array.shape[0],
               'dim-y': ct_dicom.pixel_array.shape[1],
               'offset-x': ct_dicom.ImagePositionPatient[0], 
               'offset-y': ct_dicom.ImagePositionPatient[1], 
               'offset-z': ct_dicom.ImagePositionPatient[2], 
               'res-x': ct_dicom.PixelSpacing[0],
               'res-y': ct_dicom.PixelSpacing[1],
               'scale-int': ct_dicom.RescaleIntercept,
               'scale-slope': ct_dicom.RescaleSlope,
               'hu-min': hus.min(),
               'hu-max': hus.max() 
            }
            details_df = details_df.append(row_data, ignore_index=True)

        # Set column types as 'append' crushes them.
        details_df = details_df.astype(detail_cols)

        # Sort by 'offset-z'.
        details_df = details_df.sort_values('offset-z').reset_index(drop=True)

        # TODO: Add cache writing using SummaryCache object.
        # if write_cache and not cache_was_read:
        #   # Write to the cache.

        return details_df

    def rtstruct_details(self, read_cache=True, write_cache=True):
        """
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        returns: dataframe with info for each region-of-interest.
        """
        # Define table structure.
        details_cols = {
            'roi-label': 'object'
        }
        details_df = pd.DataFrame(columns=details_cols.keys())

        rois = self.dataset.get_rtstruct(self.pat_id).StructureSetROISequence
        
        # Add info for each region-of-interest.
        for roi in rois:
            row_data = {
                'roi-label': roi.ROIName
            }
            details_df = details_df.append(row_data, ignore_index=True)

        # Set column type.
        details_df = details_df.astype(details_cols)

        # Sort by label.
        details_df = details_df.sort_values('roi-label').reset_index(drop=True)

        return details_df

    def full_summary(self, read_cache=True, write_cache=True):
        """
        read_cache: reads from cache if present.
        write_cache: writes results to cache, unless read from cache.
        returns: dataframe with info for each CT slice. 
        """
        # TODO: Add caching.
        # if read_cache:
            # Check if cache exists using SummaryCache object.
            # If present, return as is, else progress with calculation.
            # return details_df
        
        # Define table structure.
        full_summary_info = {
            'dim-x': np.uint16,
            'dim-y': np.uint16,
            'dim-z': np.uint16,
            'fov-x': 'float64',
            'fov-y': 'float64',
            'fov-z': 'float64',
            'hu-min': 'float',
            'hu-max': 'float',
            'num-empty': np.uint16,
            'offset-x': 'float',
            'offset-y': 'float',
            'res-x': 'float',
            'res-y': 'float',
            'res-z': 'float',
            'roi-num': np.uint16,
            'scale-int': 'float',
            'scale-slope': 'float'
        }