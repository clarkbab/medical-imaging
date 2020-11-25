import os
import numpy as np
import pandas as pd
from skimage.draw import polygon
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientSummary

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

    def __init__(self, pat_id, dataset=ds):
        """
        pat_id: a patient ID string.
        dataset: a DICOM dataset.
        """
        self.dataset = dataset
        self.pat_id = pat_id

    def get_data(self):
        """
        returns: a numpy array of pixel data in HU.
        """
        # Load patient CT dicoms.
        ct_dicoms = self.dataset.list_ct(self.pat_id)
        pat_sum = PatientSummary.from_id(self.pat_id, dataset=self.dataset)
        ct_details_df = pat_sum.ct_details() 

        # Ensure that CT slice dimensions are consistent.
        assert len(ct_details_df['dim-x'].unique()) == 1
        assert len(ct_details_df['dim-y'].unique()) == 1

        # Calculate 'dim-z'.
        res_zs = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_details_df['offset-z'])])
        res_z = res_zs[0]   # Take smallest resolution.
        fov_z = ct_details_df['offset-z'].max() - ct_details_df['offset-z'].min()
        dim_z = int(round(fov_z / res_z, 0)) + 1

        # Create placeholder array.
        data_shape = (ct_details_df['dim-x'][0], ct_details_df['dim-y'][0], dim_z)
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
            z_idx = int(round(offset_z / res_z))

            # Add data.
            data[:, :, z_idx] = pixel_data

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
        assert len(ct_details_df['dim-x'].unique()) == 1
        assert len(ct_details_df['dim-y'].unique()) == 1

        # Calculate res-z - this will be the smallest available diff.
        res_zs = np.sort([round(i, FLOAT_DP) for i in np.diff(ct_details_df['offset-z'])])
        res_z = res_zs[0]

        # Calculate fov-z and dim-z.
        fov_z = ct_details_df['offset-z'].max() - ct_details_df['offset-z'].min()
        dim_z = int(fov_z / res_z) + 1

        labels = []

        # Create and add labels.
        for roi, roi_info in zip(rois, roi_infos):
            # Create label placeholder.
            label_shape = (ct_details_df['dim-x'][0], ct_details_df['dim-y'][0], dim_z)
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
