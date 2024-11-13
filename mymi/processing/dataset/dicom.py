import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
from pandas import DataFrame, MultiIndex, read_csv
from pathlib import Path
import re
import shutil
import SimpleITK as sitk
import struct
from time import time
from typing import List, Optional
from tqdm import tqdm

from mymi.dataset.shared import CT_FROM_REGEXP
from mymi.dataset.dicom import DicomDataset
from mymi.dataset.nifti import NiftiDataset, recreate as recreate_nifti
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import sitk_image_transform, sitk_point_transform
from mymi.types import PatientRegions
from mymi.utils import append_row, arg_to_list, save_csv, save_nifti, to_sitk_image

from .dataset import write_flag

ERROR_COLS = {
    'error': str
}
ERROR_INDEX = [
    'dataset',
    'patient-id'
]

def convert_to_nifti(
    dataset: 'Dataset',
    anonymise: bool = False,
    regions: Optional[PatientRegions] = 'all',
    show_list_patients_progress: bool = True) -> None:
    logging.arg_log('Converting DicomDataset to NiftiDataset', ('dataset', 'anonymise', 'regions'), (dataset, anonymise, regions))
    start = time()

    # Create NIFTI dataset.
    dicom_set = DicomDataset(dataset)
    nifti_set = recreate_nifti(dataset)

    if regions == 'all':
        regions = dicom_set.list_regions()
    else:
        regions = regions_to_list(regions)

    # Check '__ct_from_' for DICOM dataset.
    ct_from = None
    for f in os.listdir(dicom_set.path):
        match = re.match(CT_FROM_REGEXP, f)
        if match:
            ct_from = match.group(1)

    # Add '__ct_from_' tag to NIFTI dataset.
    if ct_from is not None:
        filepath = os.path.join(nifti_set.path, f'__CT_FROM_{ct_from}__')
        open(filepath, 'w').close()

    # Load all patients.
    pat_ids = dicom_set.list_patients(regions=regions, show_progress=show_list_patients_progress)

    if anonymise:
        # Create CT map. Index of map will be the anonymous ID.
        df = DataFrame(pat_ids, columns=['patient-id']).reset_index().rename(columns={ 'index': 'id' })

        # Save map.
        filepath = os.path.join(dicom_set.path, 'anon-nifti-map.csv')
        save_csv(df, filepath, overwrite=True)

    # Keep track of errors - but don't let errors stop the processing.
    error_index = MultiIndex(levels=[[], []], codes=[[], []], names=ERROR_INDEX)
    error_df = DataFrame(columns=ERROR_COLS.keys(), index=error_index)

    for pat_id in tqdm(pat_ids):
        try:
            # Get anonymous ID.
            if anonymise:
                anon_id = df[df['patient-id'] == pat_id].index.values[0]
                filename = f'{anon_id}.nii.gz'
            else:
                filename = f'{pat_id}.nii.gz'

            # Create CT NIFTI.
            pat = dicom_set.patient(pat_id)
            data = pat.ct_data
            spacing = pat.ct_spacing
            offset = pat.ct_offset
            affine = np.array([
                [spacing[0], 0, 0, offset[0]],
                [0, spacing[1], 0, offset[1]],
                [0, 0, spacing[2], offset[2]],
                [0, 0, 0, 1]])
            if ct_from is None:
                img = Nifti1Image(data, affine)
                filepath = os.path.join(nifti_set.path, 'data', 'ct', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

            # Create region NIFTIs.
            region_data = pat.region_data(only=regions)
            for r, data in region_data.items():
                img = Nifti1Image(data.astype(np.int32), affine)
                filepath = os.path.join(nifti_set.path, 'data', 'regions', r, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

            # Create RTDOSE NIFTI.
            dose_data = pat.dose_data
            if dose_data is not None:
                img = Nifti1Image(dose_data, affine)
                filepath = os.path.join(nifti_set.path, 'data', 'dose', filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)
        except ValueError as e:
            data_index = [dataset, pat_id] 
            data = {
                'error': str(e)
            }
            error_df = append_row(error_df, data, index=data_index)

    # Save errors index.
    if len(error_df) > 0:
        error_df = error_df.astype(ERROR_COLS)
    filepath = os.path.join(nifti_set.path, 'conversion-errors.csv')
    error_df.to_csv(filepath, index=True)

    # Save indexing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    filepath = os.path.join(nifti_set.path, f'__CONVERSION_TIME_MINS_{mins}__')
    Path(filepath).touch()

def convert_to_nifti_replan(
    dataset: str,
    dicom_dataset: Optional[str] = None,
    region: PatientRegions = 'all',
    anonymise: bool = False) -> None:
    regions = regions_to_list(region)

    # Create NIFTI dataset.
    nifti_set = recreate_nifti(dataset)
    logging.arg_log('Converting replan dataset to NIFTI', ('dataset', 'regions', 'anonymise'), (dataset, regions, anonymise))

    # Get all patients.
    dicom_dataset = dataset if dicom_dataset is None else dicom_dataset
    set = DicomDataset(dicom_dataset)
    filepath = os.path.join(set.path, 'patient-studies.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"File '<dataset>/patient-studies.csv' not found.")
    study_df = read_csv(filepath, dtype={ 'patient-id': str })
    pat_ids = list(sorted(np.unique(study_df['patient-id'])))

    if anonymise:
        cols = {
            'patient-id': str,
            'origin-dataset': str,
            'origin-patient-id': str,
            'origin-study-id': str
        }
        df = DataFrame(columns=cols.keys())

    for i, pat_id in enumerate(tqdm(pat_ids)):
        # Get study IDs.
        study_ids = study_df[study_df['patient-id'] == pat_id]['study-id'].values

        for j, study_id in enumerate(study_ids):
            # Get ID.
            if anonymise:
                nifti_id = f'{i}-{j}'
            else:
                nifti_id = f'{pat_id}-{j}'

            # Add row to anon index.
            if anonymise:
                data = {
                    'patient-id': nifti_id,
                    'origin-dataset': dicom_dataset,
                    'origin-patient-id': pat_id,
                    'origin-study-id': study_id,
                }
                df = append_row(df, data)

            # Create CT NIFTI for study.
            pat = set.patient(pat_id)
            study = pat.study(study_id)
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            ct_offset = study.ct_offset
            affine = np.array([
                [ct_spacing[0], 0, 0, ct_offset[0]],
                [0, ct_spacing[1], 0, ct_offset[1]],
                [0, 0, ct_spacing[2], ct_offset[2]],
                [0, 0, 0, 1]])
            img = Nifti1Image(ct_data, affine)
            filepath = os.path.join(nifti_set.path, 'data', 'ct', f'{nifti_id}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

            # Create region NIFTIs for study.
            region_data = study.region_data(only=regions)
            for region, data in region_data.items():
                img = Nifti1Image(data.astype(np.int32), affine)
                filepath = os.path.join(nifti_set.path, 'data', 'regions', region, f'{nifti_id}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

            # Create RTDOSE NIFTIs for study.
            dose_data = study.dose_data
            if dose_data is not None:
                dose_spacing = study.dose_spacing
                dose_offset = study.dose_offset
                affine = np.array([
                    [dose_spacing[0], 0, 0, dose_offset[0]],
                    [0, dose_spacing[1], 0, dose_offset[1]],
                    [0, 0, dose_spacing[2], dose_offset[2]],
                    [0, 0, 0, 1]])
                img = Nifti1Image(dose_data, affine)
                filepath = os.path.join(nifti_set.path, 'data', 'dose', f'{nifti_id}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

    if anonymise:
        filepath = os.path.join(nifti_set.path, 'index.csv') 
        df.to_csv(filepath, index=False)

    # Indicate success.
    write_flag(nifti_set, '__CONVERT_FROM_NIFTI_END__')

def __convert_velocity_transform(filepath: str) -> None:
    # Read ".bdf" file.
    with open(filepath, "rb") as f:
        data = f.read()
        
    # Read transform image size.
    # Read data as "sliding windows" of bytes.
    # Data format is 32-bit unsigned int (I), little-endian (<).
    size = []
    n_dims = 3
    n_bytes_per_val = 4
    n_bytes = n_dims * n_bytes_per_val
    data_format = "<I"
    for i in range(0, n_bytes, n_bytes_per_val):
        size_i = struct.unpack(data_format, data[i:i + n_bytes_per_val])[0]
        size.append(size_i)
    size = tuple(size)

    # Read transform image pixel spacing.
    # Data format is 32-bit float (f).
    spacing = []
    data_format = "f"
    start_byte = 12
    n_dims = 3
    n_bytes = n_dims * n_bytes_per_val
    for i in range(start_byte, start_byte + n_bytes, n_bytes_per_val):
        spacing_i = struct.unpack(data_format, data[i:i + n_bytes_per_val])[0]
        spacing.append(spacing_i)
    spacing = tuple(spacing)

    # Sanity check number of bytes in file.
    # Should be num. voxels * 3 * 4 (each voxel is a 3 dimensional, 32-bit float) + 24 bytes for image size and spacing header.
    n_voxels = np.prod(size)
    n_bytes = len(data)
    n_bytes_expected = n_voxels * n_bytes_per_val * n_dims + 24
    if n_bytes != n_bytes_expected:
        raise ValueError(f"File '{filepath}' should contain '{n_bytes_expected}' bytes (num. voxels ({n_voxels}) * 4 bytes * 3 axes + 24 bytes header), got '{n_bytes}'.")

    # Read vector image.
    vector = []
    image = []
    start_byte = 24
    n_bytes = n_voxels * n_dims * n_bytes_per_val
    for i in range(start_byte, start_byte + n_bytes, n_bytes_per_val):
        vector_i = struct.unpack(data_format, data[i:i + n_bytes_per_val])[0]
        vector.append(vector_i)
        if (i - (start_byte - n_bytes_per_val)) % n_dims * n_bytes_per_val == 0:
            image.append(vector.copy())
            vector = []
            
    if len(image) != n_voxels:
        raise ValueError(f"Expected image to contain '{n_voxels}' voxels, got '{len(image)}'.")

    # When performing numpy.reshape, data is taken from the flat array for the last dimension first.
    # E.g. if reshaping to (3, 4), the first 4 elements of the flat array are taken for the first row,
    # followed by another 4 elements for the second row, etc.
    # Velocity stores data in the opposite order, e.g. the first 3 elements should be taken for the first
    # column, followed by the next 3 elements for the second column, etc.
    # To get this behaviour, we reverse the shape for numpy.reshape, e.g. (4, 3), and then move axes
    # after reshaping.
    image = np.array(image)
    image = np.reshape(image, (*size[::-1], 3))
    image = np.moveaxis(np.moveaxis(image, 0, 2), 0, 1)

    # Create transform.
    origin = (0, 0, 0)
    image = to_sitk_image(image, spacing, origin, is_vector=True)
    transform = sitk.DisplacementFieldTransform(image)
    transformpath = filepath.replace('.bdf', '.tfm')
    sitk.WriteTransform(transform, transformpath)

def convert_velocity_predictions_to_nifti(
    dataset: str,
    transform_types: List[str] = ['DMP', 'EDMP']) -> None:
    dicom_set = DicomDataset(dataset)
    nifti_set = NiftiDataset(dataset)
    pat_ids = dicom_set.list_patients()
    pat_ids = [p.split('_')[-1] for p in pat_ids]
    for p in tqdm(pat_ids):
        # Load fixed/moving images.
        fixed_pat = nifti_set.patient(f'{p}-1')
        fixed_spacing = fixed_pat.ct_spacing
        fixed_offset = fixed_pat.ct_offset
        moving_pat = nifti_set.patient(f'{p}-0')
        moving_spacing = moving_pat.ct_spacing
        moving_offset = moving_pat.ct_offset

        for t in transform_types:
            # Convert Velocity transform to sitk.
            transformpath = os.path.join(dicom_set.path, 'data', 'velocity', 'dvf', f'{p}_{t}.bdf')
            if not os.path.exists(transformpath):
                logging.info(f"Skipping prediction '{transformpath}' - not found.")
                continue
            __convert_velocity_transform(transformpath)

            # Load sitk transform.
            transformpath = transformpath.replace('.bdf', '.tfm')
            warp = sitk.ReadTransform(transformpath)

            # Apply transform to moving image.
            moved = sitk_image_transform(fixed_pat.ct_data, moving_pat.ct_data, fixed_spacing, moving_pat.ct_spacing, fixed_offset, moving_pat.ct_offset, warp)
                
            # Save prediction.
            modelname = f'VELOCITY-{t}'
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', modelname, 'ct', f'{p}-0.nii.gz')
            save_nifti(moved, fixed_spacing, fixed_offset, filepath)

            # Move transform.
            filepath = filepath.replace('.nii.gz', '_warp.tfm')
            shutil.copyfile(transformpath, filepath)

            # Apply transform to *fixed* landmarks.
            moved_lms = sitk_point_transform(fixed_pat.landmarks, fixed_spacing, moving_spacing, fixed_offset, moving_offset, warp)

            # Save transformed points.
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', modelname, 'landmarks', f'{moving_pat.id}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savetxt(filepath, moved_lms, delimiter=',', fmt='%.3f')
