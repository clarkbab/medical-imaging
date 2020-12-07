import gzip
import logging
import math
import numpy as np
import os
from skimage.draw import polygon
import shutil
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import cache
from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientDataExtractor, PatientInfo

CACHE_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'cache')
PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '2d-parotid-left')

class DatasetPreprocessor:
    def __init__(self, dataset=ds):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = dataset

    def extract(self, drop_missing_slices=True, num_pats='all', regions='all', transforms=[]):
        """
        effect: stores processed patient data.
        drop_missing_slices: drops patients that have missing slices.
        num_pats: operate on subset of patients.
        transform: apply the pre-defined transformation.
        """
        # Load patients.
        pat_ids = self.dataset.list_patients()

        # Get patient subset.
        if num_pats != 'all':
            assert isinstance(num_pats, int)
            pat_ids = pat_ids[:num_pats]

        # Maintain global sample index.
        pos_sample_idx = 0
        neg_sample_idx = 0

        # Process data for each patient.
        for pat_id in tqdm(pat_ids):
            logging.info(f"Extracting data for patient {pat_id}.")

            # Check if there are missing slices.
            pi = PatientInfo(pat_id)
            patient_info_df = pi.full_info()

            if drop_missing_slices and patient_info_df['num-missing'][0] != 0:
                logging.info(f"Dropping patient '{pat_id}' with missing slices.")
                continue

            # Load input data.
            pde = PatientDataExtractor(pat_id)
            data = pde.get_data(transforms=transforms)

            # Load label data.
            labels = pde.get_labels(regions=regions, transforms=transforms)

            for lname, ldata in labels:
                # Find slices that are labelled.
                pos_indices = ldata.sum(axis=(0, 1)).nonzero()[0]

                # Write positive input and label data.
                label_path = os.path.join(PROCESSED_ROOT, lname)
                for pos_idx in pos_indices: 
                    # Get input and label data.
                    input_data = data[:, :, pos_idx]
                    label_data = ldata[:, :, pos_idx]

                    # Save input data.
                    pos_path = os.path.join(label_path, 'positive')
                    os.makedirs(pos_path, exist_ok=True)
                    filename = f"{pos_sample_idx:05}-input"
                    filepath = os.path.join(pos_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, input_data)

                    # Save label.
                    filename = f"{pos_sample_idx:05}-label"
                    filepath = os.path.join(pos_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, label_data)

                    # Increment sample index.
                    pos_sample_idx += 1

                # Find slices that aren't labelled.
                neg_indices = np.setdiff1d(range(ldata.shape[2]), pos_indices) 

                # Write negative input and label data.
                for neg_idx in neg_indices:
                    # Get input and label data.
                    input_data = data[:, :, neg_idx]
                    label_data = ldata[:, :, neg_idx]

                    # Save input data.
                    neg_path = os.path.join(label_path, 'negative')
                    os.makedirs(neg_path, exist_ok=True)
                    filename = f"{neg_sample_idx:05}-input"
                    filepath = os.path.join(neg_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, input_data)

                    # Save label.
                    filename = f"{neg_sample_idx:05}-label"
                    filepath = os.path.join(neg_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, label_data)

                    # Increment sample index.
                    neg_sample_idx += 1

        # Write to train, validate and test folders.
        label_path = os.path.join(PROCESSED_ROOT, 'Parotid-Left')
        folders = ['positive', 'negative']
        new_folders = ['train', 'validate', 'test']

        for folder in folders:
            folder_path = os.path.join(label_path, folder)
            samples = np.sort(os.listdir(folder_path)).reshape((-1, 2))
            shuffled_idx = np.random.permutation(len(samples))

            # Get train/test/validate numbers.
            num_train = math.floor(0.6 * len(shuffled_idx))
            num_validate = math.floor(0.2 * len(shuffled_idx))
            num_test = math.floor(0.2 * len(shuffled_idx))

            ranges = [range(num_train), range(num_train, num_validate + num_train), range(num_train + num_validate, num_train + num_validate + num_test)]

            for new_folder, rnge in zip(new_folders, ranges):
                path = os.path.join(label_path, new_folder)
                print(os.path.join(path, folder))
                os.makedirs(os.path.join(path, folder), exist_ok=True)

                for input_file, label_file in samples[rnge]:
                    os.rename(os.path.join(folder_path, input_file), os.path.join(path, folder, input_file))
                    os.rename(os.path.join(folder_path, label_file), os.path.join(path, folder, label_file))

        # Clean up initial folders.
        for folder in folders:
            shutil.rmtree(os.path.join(label_path, folder))

    def get_patient_data(self, pat_id, transforms=[]):
        """
        returns: a numpy array of pixel data in HU.
        pat_id: the patient ID.
        transforms: a list of transforms to apply to the data.
        """
        key = {
            'class': 'patient_data_extractor',
            'method': 'get_data',
            'patient_id': pat_id,
            'transforms': [t.cache_id() for t in transforms]
        }
        if cache.enabled_read and cache.exists(key):
            cache.read(key, 'array')

        # Load patient CT dicoms.
        ct_dicoms = self.dataset.list_ct(pat_id)
        pi = PatientInfo(pat_id, dataset=self.dataset)
        full_info_df = pi.full_info()
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
            data = transform(data, full_info)

        # Write data to cache.
        if cache.enabled_write:
            cache.write(key, data, 'array')

        return data
                    
    def get_patient_labels(self, pat_id, regions='all', transforms=[]):
        """
        returns: a list of (<label name>, <label data>) pairs.
        pat_id: the patient ID.
        regions: the desired regions.
        transforms: a list of transforms to apply to the labels.
        """
        key = {
            'class': 'patient_data_extractor',
            'method': 'get_labels',
            'patient_id': pat_id,
            'regions': regions,
            'transforms': [t.cache_id() for t in transforms]
        }
        if cache.enabled_read and cache.exists(key):
            cache.read(key, 'name-array-pairs')

        # Load all regions-of-interest.
        rtstruct_dicom = self.dataset.get_rtstruct(pat_id)
        rois = rtstruct_dicom.ROIContourSequence
        roi_infos = rtstruct_dicom.StructureSetROISequence

        # Load CT data for label shape.
        pi = PatientInfo(pat_id, dataset=self.dataset)
        full_info_df = pi.full_info()
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
            data = np.zeros(shape=data_shape, dtype=np.bool)

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
            labels = [(name, transform(data, full_info)) for name, data in labels]

        # Write data to cache.
        if cache.enabled_write:
            cache.write(key, labels, 'name-array-pairs')

        return labels
