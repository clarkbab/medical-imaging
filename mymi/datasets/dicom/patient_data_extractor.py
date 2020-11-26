import os
import numpy as np
import pandas as pd
import scipy
from skimage.draw import polygon
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientSummary
from mymi.cache import DataCache

CACHE_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'cache')
FLOAT_DP = 2

class PatientDataExtractor:
    @staticmethod
    def from_id(pat_id, dataset=ds):
        """
        pat_id: an identifier for the patient.
        returns: PatientSummary object.
        """
        # TODO: flesh out.
        if not dataset.has_id(pat_id):
            print(f"Patient ID '{pat_id}' not found in dataset.")
            # raise error
            exit(0)

        return PatientDataExtractor(pat_id, dataset=dataset)

    def __init__(self, pat_id, dataset=ds, verbose=False):
        """
        pat_id: a patient ID string.
        dataset: a DICOM dataset.
        """
        self.cache = DataCache(CACHE_ROOT)
        self.dataset = dataset
        self.pat_id = pat_id
        self.verbose = verbose

    def get_data(self, read_cache=True, write_cache=True, transform=False):
        """
        returns: a numpy array of pixel data in HU.
        read_cache: reads from cache if present.
        resolution: the resampling resolution. 
        write_cache: writes results to cache, unless read from cache.
        """
        key = {
            'class': 'patient_data_extractor',
            'method': 'get_data',
            'patient_id': self.pat_id,
            'transform': transform,
        }
        if read_cache:
            if self.cache.exists(key):
                if self.verbose: print(f"Reading cache for key {key}.")
                return self.cache.read(key, 'array')

        # Load patient CT dicoms.
        ct_dicoms = self.dataset.list_ct(self.pat_id)
        pat_sum = PatientSummary(self.pat_id, dataset=self.dataset)
        ct_details_df = pat_sum.ct_details(read_cache=read_cache, write_cache=write_cache) 

        # Ensure that CT slice dimensions are consistent.
        assert len(ct_details_df['res-x'].unique()) == 1
        assert len(ct_details_df['res-y'].unique()) == 1

        # Calculate 'res-z'.
        spacing_zs = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_details_df['offset-z'])])
        spacing_z = spacing_zs[0]   # Take smallest resolution.
        fov_z = ct_details_df['offset-z'].max() - ct_details_df['offset-z'].min()
        res_z = int(round(fov_z / spacing_z, 0)) + 1

        # Create placeholder array.
        data_shape = (ct_details_df['res-x'][0], ct_details_df['res-y'][0], res_z)
        data = np.zeros(shape=data_shape)
        
        # Add CT data.
        for ct_dicom in ct_dicoms:
            # Convert stored data to HU.
            pixel_data = ct_dicom.pixel_array
            pixel_data = ct_dicom.RescaleSlope * pixel_data + ct_dicom.RescaleIntercept

            # Transpose to put in the form (x, y) where x is the table axis.
            pixel_data = np.transpose(pixel_data)

            # Get z index.
            offset_z =  ct_dicom.ImagePositionPatient[2] - ct_details_df['offset-z'][0]
            z_idx = int(round(offset_z / spacing_z))

            # Add data.
            data[:, :, z_idx] = pixel_data


        # Perform resampling.
        if transform:
            # Ensure that CT slice resolutions are consisent.
            assert len(ct_details_df['spacing-x'].unique()) == 1
            assert len(ct_details_df['spacing-y'].unique()) == 1

            new_spacing = (0.976562, 0.976562, 3.0)     # TODO: pass in.
            old_spacing = np.array([ct_details_df['spacing-x'][0], ct_details_df['spacing-y'][0], spacing_z])

            # No resampling to be performed.
            if np.array_equal(new_spacing, old_spacing):
                # Write data to cache.
                if write_cache:
                    if self.verbose: print(f"Writing cache key {key}.")
                    self.cache.write(key, data, 'array') 
            
                return data

            # Calculate shape resize factor - the ratio of new to old pixel numbers.
            resize_factor = old_spacing / new_spacing
            print(old_spacing)

            # Calculate new shape - rounded to nearest integer.
            new_shape = np.round(data.shape * resize_factor)

            # Our real spacing will be different from 'new spacing' due to shape
            # consisting of integers. The field-of-view (shape * spacing) must be
            # maintained throughout.
            real_resize_factor = new_shape / data.shape
            new_spacing = old_spacing / real_resize_factor

            # Perform resampling.
            data = scipy.ndimage.zoom(data, real_resize_factor)

        # Write data to cache.
        if write_cache:
            if self.verbose: print(f"Writing cache key {key}.")
            self.cache.write(key, data, 'array') 

        return data

    def list_labels(self):
        """
        returns: a list of (<label name>, <label data>) pairs.
        """
        # Load all regions-of-interest.
        rtstruct_dicom = self.dataset.get_rtstruct(self.pat_id)
        rois = rtstruct_dicom.ROIContourSequence
        roi_infos = rtstruct_dicom.StructureSetROISequence

        # Load CT data for label shape.
        pat_sum = PatientSummary.from_id(self.pat_id, dataset=self.dataset)
        ct_details_df = pat_sum.ct_details()

        # Check for consistency among scans.
        assert len(ct_details_df['res-x'].unique()) == 1
        assert len(ct_details_df['res-y'].unique()) == 1

        # Calculate spacing-z - this will be the smallest available diff.
        spacing_zs = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_details_df['offset-z'])])
        spacing_z = spacing_zs[0]

        # Calculate fov-z and res-z.
        fov_z = ct_details_df['offset-z'].max() - ct_details_df['offset-z'].min()
        res_z = int(fov_z / spacing_z) + 1

        labels = []

        # Create and add labels.
        for roi, roi_info in zip(rois, roi_infos):
            # Create label placeholder.
            label_shape = (ct_details_df['res-x'][0], ct_details_df['res-y'][0], res_z)
            label = np.zeros(shape=label_shape, dtype=np.uint8)

            roi_coords = [c.ContourData for c in roi.ContourSequence]

            # Label each slice of the ROI.
            for roi_slice_coords in roi_coords:
                # Coords are stored in flat array.
                coords = np.array(roi_slice_coords).reshape(-1, 3)

                # Convert from "real" space to pixel space using affine transformation.
                corner_pixels_x = (coords[:, 0] - ct_details_df['offset-x'].min()) / ct_details_df['res-x'][0]
                corner_pixels_y = (coords[:, 1] - ct_details_df['offset-y'].min()) / ct_details_df['res-y'][0]

                # Get contour z pixel.
                offset_z = coords[0, 2] - ct_details_df['offset-z'].min()
                pixel_z = int(offset_z / res_z)

                # Get 2D coords of polygon boundary and interior described by corner
                # points.
                pixels_x, pixels_y = polygon(corner_pixels_x, corner_pixels_y)

                # Set labelled pixels in slice.
                label[pixels_x, pixels_y, pixel_z] = 1

            labels.append((roi_info.ROIName, label))

        # Sort by label name.
        labels = sorted(labels, key=lambda l: l[0])

        return labels
