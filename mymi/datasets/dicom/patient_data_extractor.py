import os
import numpy as np
import pandas as pd
import scipy
from skimage.draw import polygon
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientInfo
from mymi.cache import DataCache

CACHE_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'cache')
FLOAT_DP = 2

class PatientDataExtractor:
    def __init__(self, pat_id, dataset=ds, verbose=False):
        """
        pat_id: a patient ID string.
        dataset: a DICOM dataset.
        """
        self.cache = DataCache(CACHE_ROOT, verbose=verbose)
        self.dataset = dataset
        self.pat_id = pat_id
        self.verbose = verbose

    def get_data(self, read_cache=True, transforms=[], write_cache=True):
        """
        returns: a numpy array of pixel data in HU.
        read_cache: reads from cache if present.
        transforms: a list of transforms to apply to the data.
        write_cache: writes results to cache, unless read from cache.
        """
        key = {
            'class': 'patient_data_extractor',
            'method': 'get_data',
            'patient_id': self.pat_id,
            'transforms': [t.cache_id() for t in transforms]
        }
        if read_cache:
            if self.cache.exists(key):
                return self.cache.read(key, 'array')

        # Load patient CT dicoms.
        ct_dicoms = self.dataset.list_ct(self.pat_id)
        pi = PatientInfo(self.pat_id, dataset=self.dataset, verbose=self.verbose)
        full_info_df = pi.full_info(read_cache=read_cache, write_cache=write_cache)
        full_info = full_info_df.iloc[0].to_dict()

        # Create placeholder array.
        data_shape = (int(full_info['res-x']), int(full_info['res-y']), int(full_info['res-z']))
        data = np.zeros(shape=data_shape, dtype=np.int16)
        
        # Add CT data.
        for ct_dicom in ct_dicoms:
            # Convert stored data to HU.
            pixel_data = ct_dicom.pixel_array
            pixel_data = ct_dicom.RescaleSlope * pixel_data + ct_dicom.RescaleIntercept

            # Transpose to put in the form (x, y) where x is the table axis.
            pixel_data = np.transpose(pixel_data)

            # Get z index.
            offset_z =  ct_dicom.ImagePositionPatient[2] - full_info['offset-z']
            z_idx = int(round(offset_z / full_info['spacing-z']))

            # Add data.
            data[:, :, z_idx] = pixel_data

        # Transform the data.
        for transform in transforms:
            data = transform.run(data, full_info)

        # Write data to cache.
        if write_cache:
            self.cache.write(key, data, 'array') 

        return data

    def get_labels(self, read_cache=True, regions='all', transforms=[], write_cache=True):
        """
        returns: a list of (<label name>, <label data>) pairs.
        regions: the desired regions.
        read_cache: reads from cache if present.
        transforms: a list of transforms to apply to the labels.
        write_cache: writes results to cache, unless read from cache.
        """
        key = {
            'class': 'patient_data_extractor',
            'method': 'get_labels',
            'patient_id': self.pat_id,
            'regions': regions,
            'transforms': [t.cache_id() for t in transforms]
        }
        if read_cache:
            if self.cache.exists(key):
                return self.cache.read(key, 'name-array-pairs')

        # Load all regions-of-interest.
        rtstruct_dicom = self.dataset.get_rtstruct(self.pat_id)
        rois = rtstruct_dicom.ROIContourSequence
        roi_infos = rtstruct_dicom.StructureSetROISequence

        # Load CT data for label shape.
        pi = PatientInfo(self.pat_id, dataset=self.dataset, verbose=self.verbose)
        full_info_df = pi.full_info(read_cache=read_cache, write_cache=write_cache)
        full_info = full_info_df.iloc[0].to_dict()

        labels = []

        # Create and add labels.
        for roi, roi_info in zip(rois, roi_infos):
            name = roi_info.ROIName

            # Check if we should skip.
            if not (regions == 'all' or
                (type(regions) == list and name in regions) or
                (type(regions) == str and name == regions)):
                continue

            # Create label placeholder.
            data_shape = (int(full_info['res-x']), int(full_info['res-y']), int(full_info['res-z']))
            data = np.zeros(shape=data_shape, dtype=np.uint8)

            roi_coords = [c.ContourData for c in roi.ContourSequence]

            # Label each slice of the ROI.
            for roi_slice_coords in roi_coords:
                # Coords are stored in flat array.
                coords = np.array(roi_slice_coords).reshape(-1, 3)

                # Convert from "real" space to pixel space using affine transformation.
                corner_pixels_x = (coords[:, 0] - full_info['offset-x']) / full_info['spacing-x']
                corner_pixels_y = (coords[:, 1] - full_info['offset-y']) / full_info['spacing-y']

                # Get contour z pixel.
                offset_z = coords[0, 2] - full_info['offset-z']
                pixel_z = int(offset_z / full_info['spacing-z'])

                # Get 2D coords of polygon boundary and interior described by corner
                # points.
                pixels_x, pixels_y = polygon(corner_pixels_x, corner_pixels_y)

                # Set labelled pixels in slice.
                data[pixels_x, pixels_y, pixel_z] = 1

            labels.append((name, data))

        # Sort by label name.
        labels = sorted(labels, key=lambda l: l[0])

        # Transform the labels.
        full_info['order'] = 0      # Perform nearest-neighbour interpolation.
        for transform in transforms:
            labels = [(name, transform.run(data, full_info)) for name, data in labels]

        # Write data to cache.
        if write_cache:
            self.cache.write(key, labels, 'name-array-pairs') 

        return labels
