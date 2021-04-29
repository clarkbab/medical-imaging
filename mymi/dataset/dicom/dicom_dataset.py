import inspect
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydicom as pdcm
from scipy.ndimage import center_of_mass
from skimage.draw import polygon
from torchio import ScalarImage, Subject
from tqdm import tqdm

from mymi import cache
from mymi import dataset
from mymi.utils import filterOnNumPats, filterOnPatID, filterOnLabel, stringOrSorted

Z_SPACING_ROUND_DP = 2

class DicomDataset:
    # Subclasses must implement.

    @classmethod
    def clinical_data(cls):
        raise NotImplementedError("Method 'clinical_data' not implemented in subclass.")

    @classmethod
    def data_dir(cls):
        raise NotImplementedError("Method 'data_dir' not implemented in subclass.")

    # Queries.

    @classmethod
    def generate_report(cls, name, label='all'):
        """
        effect: generates a PDF report for the dataset.
        args:
            name: the report name.
        kwargs:
            label: include patients with any of the listed labels (behaves like an OR).
        """
        # Load patient summaries.
        summary = dataset.patient_summaries(label=label)

        # Create CT section.

        # Plot acquisition parameters and show table of outliers.
        cols = ['spacing-x', 'spacing-y', 'spacing-z']
        cols = ['size-x', 'size-y', 'size-z']
        cols = ['fov-x', 'fov-y', 'fov-z']
        
        # Display central 'sagittal' slice of those 5 patients that have smallest 'fov-z'.

        # Plot CT HU distribution. 

    @classmethod
    def has_id(cls, pat_id):
        return True

    @classmethod
    def list_ct(cls, pat_id):
        """
        returns: a list of CT dicoms.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(cls.data_dir(), pat_id)
        date_path = os.path.join(pat_path, os.listdir(pat_path)[0])
        dicom_paths = [os.path.join(date_path, p) for p in os.listdir(date_path)]

        # Find first folder containing CT scans.
        for p in dicom_paths:
            file_path = os.path.join(p, os.listdir(p)[0])
            dicom = pdcm.read_file(file_path)
            
            if dicom.Modality == 'CT':
                ct_dicoms = [pdcm.read_file(os.path.join(p, d)) for d in os.listdir(p)]
                ct_dicoms = sorted(ct_dicoms, key=lambda d: float(d.ImagePositionPatient[2]))
                return ct_dicoms

        # TODO: raise an error.
        return None

    @classmethod
    def list_patients(cls):
        """
        returns: a list of patient IDs.
        """
        return sorted(os.listdir(cls.data_dir()))

    @classmethod
    def get_rtstruct(cls, pat_id):
        """
        returns: the RTSTRUCT dicom.
        pat_id: the patient ID string.
        """
        # Get dated subfolder path.
        pat_path = os.path.join(cls.data_dir(), pat_id)
        date_path = os.path.join(pat_path, os.listdir(pat_path)[0])
        dicom_paths = [os.path.join(date_path, p) for p in os.listdir(date_path)]

        # Find first folder containing CT scans.
        for p in dicom_paths:
            file_path = os.path.join(p, os.listdir(p)[0])
            dicom = pdcm.read_file(file_path)
            
            if dicom.Modality == 'RTSTRUCT':
                return dicom

        # TODO: raise an error.
        return None

    ###
    # Raw patient data.
    ###

    @classmethod
    def patient_ct_data(cls, pat_id):
        """
        returns: a numpy array of CT data in HU.
        args:
            pat_id: the patient ID.
        """
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'pat_id': pat_id
            }
        }
        result = cache.read(params, 'array')
        if result is not None:
            return result

        # Load patient CT dicoms.
        ct_dicoms = dataset.list_ct(pat_id)
        summary = dataset.patient_ct_summary(pat_id).iloc[0].to_dict()

        # Create placeholder array.
        data_shape = (int(summary['size-x']), int(summary['size-y']), int(summary['size-z']))
        data = np.zeros(shape=data_shape, dtype=np.int16)
        
        # Add CT data.
        for ct_dicom in ct_dicoms:
            # Convert stored data to HU. Transpose to anatomical co-ordinates as
            # pixel data is stored image row-first.
            pixel_data = np.transpose(ct_dicom.pixel_array)
            pixel_data = ct_dicom.RescaleSlope * pixel_data + ct_dicom.RescaleIntercept

            # Get z index.
            offset_z =  ct_dicom.ImagePositionPatient[2] - summary['offset-z']
            z_idx = int(round(offset_z / summary['spacing-z']))

            # Add data.
            data[:, :, z_idx] = pixel_data

        # Write data to cache.
        cache.write(params, data, 'array')

        return data

    @classmethod
    def patient_label_data(cls, pat_id, label='all'):
        """
        returns: a list of (<label name>, <label data>) pairs.
        args:
            pat_id: the patient ID.
        kwargs:
            label: the desired labels.
        """
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'pat_id': pat_id
            },
            'kwargs': {
                'label': label,
            }
        }
        result = cache.read(params, 'name-array-pairs')
        if result is not None:
            return result

        # Load all labels.
        rtstruct_dicom = dataset.get_rtstruct(pat_id)
        rois = rtstruct_dicom.ROIContourSequence
        roi_infos = rtstruct_dicom.StructureSetROISequence

        # Load CT data for label shape.
        summary_df = dataset.patient_ct_summary(pat_id)
        summary = summary_df.iloc[0].to_dict()

        # Create and add labels.
        labels = []
        for roi, roi_info in zip(rois, roi_infos):
            name = roi_info.ROIName

            # Check if we should skip.
            if not (label == 'all' or
                ((type(label) == tuple or type(label) == list) and name in label) or
                (type(label) == str and name == label)):
                continue

            # Create label placeholder.
            data_shape = (int(summary['size-x']), int(summary['size-y']), int(summary['size-z']))
            data = np.zeros(shape=data_shape, dtype=np.bool)

            roi_coords = [c.ContourData for c in roi.ContourSequence]

            # Label each slice of the ROI.
            for roi_slice_coords in roi_coords:
                # Coords are stored in flat array.
                coords = np.array(roi_slice_coords).reshape(-1, 3)

                # Convert from "real" space to pixel space using affine transformation.
                corner_pixels_x = (coords[:, 0] - summary['offset-x']) / summary['spacing-x']
                corner_pixels_y = (coords[:, 1] - summary['offset-y']) / summary['spacing-y']

                # Get contour z pixel.
                offset_z = coords[0, 2] - summary['offset-z']
                pixel_z = int(offset_z / summary['spacing-z'])

                # Get 2D coords of polygon boundary and interior described by corner
                # points.
                pixels_x, pixels_y = polygon(corner_pixels_x, corner_pixels_y)

                # Set labelled pixels in slice.
                data[pixels_x, pixels_y, pixel_z] = 1

            labels.append((name, data))

        # Sort by label name.
        labels = sorted(labels, key=lambda l: l[0])

        # Write data to cache.
        cache.write(params, labels, 'name-array-pairs')

        return labels

    ###
    # Summaries of data.
    ###

    @classmethod
    def ct_summaries(cls, num_pats='all', pat_id='all', label='all'):
        """
        returns: a dataframe containing rows of patient summaries.
        kwargs:
            num_pats: the number of patients to summarise.
            pat_id: only include patients who are listed.
            label: only include patients who have at least one of the listed labels (behaves like an OR).
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'kwargs': {
                'num_pats': num_pats,
                'pat_id': stringOrSorted(pat_id),
                'label': stringOrSorted(label)
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result
                
        # Define table structure.
        cols = {
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
            'size-x': np.uint16,
            'size-y': np.uint16,
            'size-z': np.uint16,
            'spacing-x': 'float64',
            'spacing-y': 'float64',
            'spacing-z': 'float64',
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        df = pd.DataFrame(columns=cols.keys())

        # List patients.
        pat_ids = dataset.list_patients()

        # Filter patients.
        pat_ids = list(filter(filterOnPatID(pat_id), pat_ids))
        pat_ids = list(filter(filterOnLabel(label), pat_ids))
        pat_ids = list(filter(filterOnNumPats(num_pats), pat_ids))

        # Add patient info.
        for pat_id in tqdm(pat_ids):
            patient_df = dataset.patient_ct_summary(pat_id)
            patient_df['pat-id'] = pat_id
            df = df.append(patient_df)

        # Set column type.
        df = df.astype(cols)
        
        # Set index.
        df = df.set_index('pat-id')

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    @classmethod
    def patient_ct_summary(cls, pat_id):
        """
        returns: dataframe with single row summary of CT images.
        args:
            pat_id: the patient ID.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'pat_id': pat_id
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result

        # Define table structure.
        cols = {
            'fov-x': 'float64',
            'fov-y': 'float64',
            'fov-z': 'float64',
            'hu-min': 'float64',
            'hu-max': 'float64',
            'num-missing': np.uint16,
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'size-x': np.uint16,
            'size-y': np.uint16,
            'size-z': np.uint16,
            'spacing-x': 'float64',
            'spacing-y': 'float64',
            'spacing-z': 'float64',
            'scale-int': 'float64',
            'scale-slope': 'float64'
        }
        df = pd.DataFrame(columns=cols.keys())

        # Get patient scan info.
        ct_df = cls.patient_ct_slice_summary(pat_id)

        # Check for consistency among scans.
        assert len(ct_df['size-x'].unique()) == 1
        assert len(ct_df['size-y'].unique()) == 1
        assert len(ct_df['offset-x'].unique()) == 1
        assert len(ct_df['offset-y'].unique()) == 1
        assert len(ct_df['spacing-x'].unique()) == 1
        assert len(ct_df['spacing-y'].unique()) == 1
        assert len(ct_df['scale-int'].unique()) == 1
        assert len(ct_df['scale-slope'].unique()) == 1

        # Calculate spacing-z - this will be the smallest available diff.
        spacings_z = np.sort([round(i, Z_SPACING_ROUND_DP) for i in np.diff(ct_df['offset-z'])])
        spacing_z = spacings_z[0]

        # Calculate fov-z and size-z.
        fov_z = ct_df['offset-z'].max() - ct_df['offset-z'].min()
        size_z = int(round(fov_z / spacing_z, 0) + 1)

        # Calculate number of empty slices.
        num_slices = len(ct_df)
        num_missing = size_z - num_slices

        # Add table row.
        data = {
            'fov-x': ct_df['size-x'][0] * ct_df['spacing-x'][0],
            'fov-y': ct_df['size-y'][0] * ct_df['spacing-y'][0],
            'fov-z': size_z * spacing_z,
            'hu-min': ct_df['hu-min'].min(),
            'hu-max': ct_df['hu-max'].max(),
            'num-missing': num_missing,
            'offset-x': ct_df['offset-x'][0],
            'offset-y': ct_df['offset-y'][0],
            'offset-z': ct_df['offset-z'][0],
            'size-x': ct_df['size-x'][0],
            'size-y': ct_df['size-y'][0],
            'size-z': size_z,
            'spacing-x': ct_df['spacing-x'][0],
            'spacing-y': ct_df['spacing-y'][0],
            'spacing-z': spacing_z, 
            'scale-int': ct_df['scale-int'][0],
            'scale-slope': ct_df['scale-slope'][0],
        }
        df = df.append(data, ignore_index=True)

        # Set type.
        df = df.astype(cols)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    @classmethod
    def patient_ct_slice_summary(cls, pat_id):
        """
        returns: dataframe with rows containing CT slice info.
        args:
            pat_id: the patient ID.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'pat_id': pat_id
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result
            
        # Define dataframe structure.
        cols = {
            'hu-min': 'float64',
            'hu-max': 'float64',
            'offset-x': 'float64',
            'offset-y': 'float64',
            'offset-z': 'float64',
            'size-x': np.uint16,
            'size-y': np.uint16,
            'scale-int': 'float64',
            'scale-slope': 'float64',
            'spacing-x': 'float64',
            'spacing-y': 'float64',
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load CT DICOMS.
        ct_dicoms = dataset.list_ct(pat_id)
        
        # Add info.
        for ct_dicom in ct_dicoms:
            # Perform scaling from stored values to HU.
            hus = ct_dicom.pixel_array * ct_dicom.RescaleSlope + ct_dicom.RescaleIntercept

            data = {
               'hu-min': hus.min(),
               'hu-max': hus.max(),
               'offset-x': ct_dicom.ImagePositionPatient[0], 
               'offset-y': ct_dicom.ImagePositionPatient[1], 
               'offset-z': ct_dicom.ImagePositionPatient[2], 
               'size-x': ct_dicom.pixel_array.shape[1],  # Pixel array is stored (y, x) for plotting.
               'size-y': ct_dicom.pixel_array.shape[0],
               'scale-int': ct_dicom.RescaleIntercept,
               'scale-slope': ct_dicom.RescaleSlope,
               'spacing-x': ct_dicom.PixelSpacing[0],
               'spacing-y': ct_dicom.PixelSpacing[1]
            }
            df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        # Sort by 'offset-z'.
        df = df.sort_values('offset-z').reset_index(drop=True)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    @classmethod
    def labels(cls, num_pats='all', pat_id='all'):
        """
        returns: a dataframe linking patients to contoured labels.
        kwargs:
            num_pats: number of patients to summarise.
            pat_id: a string or list of patient IDs.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'kwargs': {
                'num_pats': num_pats,
                'pat_id': stringOrSorted(pat_id)
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result
                
        # Define table structure.
        cols = {
            'patient-id': 'object',
            'label': 'object'
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pat_ids = dataset.list_patients()

        # Filter patients.
        pat_ids = list(filter(filterOnPatID(pat_id), pat_ids))
        pat_ids = list(filter(filterOnNumPats(num_pats), pat_ids))

        for pat_id in tqdm(pat_ids):
            # Get rtstruct info.
            label_df = cls.patient_labels(pat_id)

            # Add rows.
            for _, row in label_df.iterrows():
                data = {
                    'patient-id': pat_id,
                    'label': row['label']
                }
                df = df.append(data, ignore_index=True)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    @classmethod
    def patient_labels(cls, pat_id, clear_cache=False):
        """
        returns: dataframe with row for each label present for the patient.
        args:
            pat_id: the patient ID.
        kwargs:
            clear_cache: whether to clear the cache or not.
        """
        # Create cache params.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'args': {
                'pat_id': pat_id
            }
        }

        # Clear cache.
        if clear_cache:
            cache.delete(params)
        
        # Read from cache.
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result
        
        # Define table structure.
        cols = {
            'label': 'object',
            'com-x': np.uint16,
            'com-y': np.uint16,
            'com-z': np.uint16,
            'width-x': np.uint16,
            'width-y': np.uint16,
            'width-z': np.uint16
        }
        df = pd.DataFrame(columns=cols.keys())

        # Get label data.
        label_data = dataset.patient_label_data(pat_id)
        
        # Add info for each label.
        for name, ldata in label_data:
            # Find centre-of-mass.
            coms = np.round(center_of_mass(ldata)).astype(np.uint16)

            # Find bounding box co-ordinates.
            non_zero = np.argwhere(ldata != 0)
            mins = non_zero.min(axis=0)
            maxs = non_zero.max(axis=0)
            widths = maxs - mins

            data = {
                'label': name,
                'com-x': coms[0],
                'com-y': coms[1],
                'com-z': coms[2],
                'width-x': widths[0],
                'width-y': widths[1],
                'width-z': widths[2]
            }
            df = df.append(data, ignore_index=True)

        # Set column type.
        df = df.astype(cols)

        # Sort by label.
        df = df.sort_values('label').reset_index(drop=True)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    @classmethod
    def label_count(cls, num_pats='all'):
        """
        returns: a dataframe containing labels and num patients with label.
        kwargs:
            num_pats: the number of patients to summarise.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'kwargs': {
                'num_pats': num_pats
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result

        # Define table structure.
        cols = {
            'num-patients': np.uint16,
            'label': 'object'
        }
        df = pd.DataFrame(columns=cols.keys())

        # List patients.
        pat_ids = dataset.list_patients()

        # Filter patients.
        pat_ids = list(filter(filterOnNumPats(num_pats), pat_ids))

        for pat_id in tqdm(pat_ids):
            # Get RTSTRUCT info.
            label_df = cls.patient_labels(pat_id=pat_id)

            # Add label counts.
            label_df['num-patients'] = 1
            df = df.merge(label_df, how='outer', on='label')
            df['num-patients'] = (df['num-patients_x'].fillna(0) + df['num-patients_y'].fillna(0)).astype(np.uint16)
            df = df.drop(['num-patients_x', 'num-patients_y'], axis=1)

        # Sort by 'roi-label'.
        df = df.sort_values('label').reset_index(drop=True)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df

    @classmethod
    def ct_statistics(cls, label='all'):
        """
        returns: a dataframe of CT statistics for the entire dataset.
        kwargs:
            label: only include data for patients with the label.
        """
        # Load from cache if present.
        params = {
            'class': cls.__name__,
            'method': inspect.currentframe().f_code.co_name,
            'kwargs': {
                'label': label
            }
        }
        result = cache.read(params, 'dataframe')
        if result is not None:
            return result

        # Convert 'labels'.
        if isinstance(label, str) and label != 'all':
            label = [label]

        # Define dataframe structure.
        cols = {
            'hu-mean': 'float64',
            'hu-std-dev': 'float64'
        }
        df = pd.DataFrame(columns=cols.keys())

        # Get patients IDs.
        pat_ids = cls.list_patients()

        # Calculate mean.
        total = 0
        num_voxels = 0
        for pat_id in pat_ids:
            # Get patient labels.
            pat_labels = list(dataset.patient_labels(pat_id)['label'])
            
            # Skip if patient has no labels, or doesn't have the specified labels.
            if len(pat_labels) == 0 or (label != 'all' and not np.array_equal(np.intersect1d(label, pat_labels), label)):
                continue

            # Add data for this patient.
            data = dataset.patient_data(pat_id)
            total += data.sum()
            num_voxels += data.shape[0] * data.shape[1] * data.shape[2]
        mean = total / num_voxels

        # Calculate standard dev.
        total = 0
        for pat_id in pat_ids:
            # Add data for the patient.
            data = dataset.patient_data(pat_id)
            total += ((data - mean) ** 2).sum()
        std_dev = np.sqrt(total / num_voxels)

        # Add data.
        data = {
            'hu-mean': mean,
            'hu-std-dev': std_dev
        }
        df = df.append(data, ignore_index=True)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        # Write data to cache.
        cache.write(params, df, 'dataframe')

        return df
