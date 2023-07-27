import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
from pandas import DataFrame, MultiIndex, read_csv
from pathlib import Path
import re
from time import time
from typing import Optional
from tqdm import tqdm

from mymi.dataset.shared import CT_FROM_REGEXP
from mymi.dataset.dicom import DICOMDataset
from mymi.dataset.nifti import recreate as recreate_nifti
from mymi import logging
from mymi.regions import region_to_list
from mymi.types import PatientRegions
from mymi.utils import append_row, arg_to_list, save_csv

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
    region: PatientRegions,
    anonymise: bool = False,
    show_list_patients_progress: bool = True) -> None:
    start = time()
    logging.info(f"Converting DICOMDataset '{dataset}' to NIFTIDataset '{dataset}', with region '{region}' and anonymise '{anonymise}'.")

    # Create NIFTI dataset.
    dicom_set = DICOMDataset(dataset)
    nifti_set = recreate_nifti(dataset)

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
    pat_ids = dicom_set.list_patients(region=region, show_progress=show_list_patients_progress)

    if anonymise:
        # Create CT map. Index of map will be the anonymous ID.
        df = DataFrame(pat_ids, columns=['patient-id']).reset_index().rename(columns={ 'index': 'anon-id' })

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
            region_data = pat.region_data(only=region)
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

def convert_to_nifti_multiple_studies(
    dataset: str,
    dicom_dataset: Optional[str] = None,
    region: PatientRegions = 'all',
    anonymise: bool = False) -> None:
    regions = region_to_list(region)

    # Create NIFTI dataset.
    nifti_set = recreate_nifti(dataset)
    logging.arg_log('Converting dataset to NIFTI', ('dataset', 'regions', 'anonymise'), (dataset, regions, anonymise))

    # Get all patients.
    dicom_dataset = dataset if dicom_dataset is None else dicom_dataset
    set = DICOMDataset(dicom_dataset)
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
        filepath = os.path.join(nifti_set.path, 'anon-index.csv') 
        df.to_csv(filepath, index=False)

    # Indicate success.
    write_flag(nifti_set, '__CONVERT_FROM_NIFTI_END__')
