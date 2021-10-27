import numpy as np
import pandas as pd
import pydicom as dcm
import os
import shutil
from tqdm import tqdm
from typing import Callable

from mymi import logging
from mymi import types

def require_hierarchy(fn: Callable) -> Callable:
    """
    effect: returns a wrapped function, ensuring hierarchy has been built.
    args:
        fn: the wrapped function.
    """
    def _require_hierarchy_wrapper(self, *args, **kwargs):
        if not _hierarchy_exists(self):
            _build_dataset_hierarchy(self)
            _trim_dataset_hierarchy(self)
        return fn(self, *args, **kwargs)
    return _require_hierarchy_wrapper

def _hierarchy_exists(
    dataset: 'DICOMDataset') -> None:
    path = os.path.join(dataset.path, 'hierarchy')
    return os.path.exists(path)

def _rollback_hierarchy(fn: Callable) -> Callable:
    def _rollback_hierarchy_wrapper(dataset: 'DICOMDataset'):
        try:
           return fn(dataset)
        except Exception as e:
            # Roll back in case of exceptions.
            path = os.path.join(dataset.path, 'hierarchy')
            if os.path.exists(path):
                logging.info(f"Rolling back hierarchy creation for dataset '{dataset}'..")
                shutil.rmtree(path)
            raise e
    return _rollback_hierarchy_wrapper

@_rollback_hierarchy
def _build_dataset_hierarchy(
    dataset: 'DICOMDataset') -> None:
    """
    effect: walks folder structure and finds DICOM files, organising them into a suitable
        hierarchy. Organising into a hierarchy allows us to interact with the data in a 
        consistent manner, regardless of the original DICOM folder structure.
    """
    # Load all dicom files.
    raw_path = os.path.join(dataset.path, 'raw')
    if not os.path.exists(raw_path):
        raise ValueError(f"No 'raw' folder found for dataset '{dataset}'.")
    dicom_files = []
    for root, _, files in os.walk(raw_path):
        for f in files:
            if f.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, f))

    # Copy dicom files.
    logging.info(f"Building hierarchy for dataset '{dataset}'..")
    for f in tqdm(dicom_files):
        # Get patient ID.
        dicom = dcm.read_file(f)
        pat_id = dicom.PatientID

        # Get modality.
        mod = dicom.Modality.lower()
        if not mod in ('ct', 'rtstruct'):
            continue

        # Get series UID.
        series_UID = dicom.SeriesInstanceUID

        # Create filepath.
        filename = os.path.basename(f)
        filepath = os.path.join(dataset.path, 'hierarchy', 'data', pat_id, mod, series_UID, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save dicom.
        dicom.save_as(filepath)

@_rollback_hierarchy
def _trim_dataset_hierarchy(
    dataset: 'DICOMDataset') -> None:
    """
    effect: removes patients with invalid data.
    """
    logging.info(f"Trimming patients with invalid data from hierarchy for dataset '{dataset}'..")

    # Create summary.
    cols = {
        'patient-id': str,
        'reason': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Get patients.
    pats = dataset.list_patients()

    # Check each patient for errors.
    for pat in tqdm(pats):
        try:
            # ValueError is thrown if:
            #   - patient is missing an RTSTRUCT file.
            #   - patient is missing CT series referenced in RTSTRUCT.
            patient = dataset.patient(pat)
        except ValueError as e:
            _trim_patient(dataset, pat, str(e))
            data = {
                'patient-id': pat,
                'reason': 'NO-VALID-SERIES'
            }
            df = df.append(data, ignore_index=True)
            continue

        # Load CT slices.
        cts = patient.get_cts()

        # Check for missing slices - i.e. instance numbers increase monotonically.
        ins = list(sorted([int(ct.InstanceNumber) for ct in cts]))
        ins_diff = np.unique(np.diff(ins))
        if len(ins_diff) != 1 or ins_diff[0] != 1:
            msg = f"Non-contiguous CT slice 'InstanceNumber' found for patient '{patient}'." 
            _trim_patient(dataset, pat, msg)
            data = {
                'patient-id': pat,
                'reason': 'NON-CONTIGUOUS-SLICES'
            }
            df = df.append(data, ignore_index=True)
            continue

        # Check for standard orientation.
        ori = cts[0].ImageOrientationPatient
        if ori != [1, 0, 0, 0, 1, 0]:
            msg = f"Non-standard CT slice 'ImageOrientationPatient' found for patient '{patient}'."
            _trim_patient(dataset, pat, msg)
            data = {
                'patient-id': pat,
                'reason': 'NON-STANDARD-ORIENTATION'
            }
            df = df.append(data, ignore_index=True)
            continue

        # Check for orientation consistency.
        error = False
        for ct in cts:
            if ct.ImageOrientationPatient != ori:
                error = True
        if error:
            msg = f"Inconsistent CT slice 'ImageOrientationPatient' found for patient '{patient}'."
            _trim_patient(dataset, pat, msg)
            data = {
                'patient-id': pat,
                'reason': 'INCONSISTENT-ORIENTATION'
            }
            df = df.append(data, ignore_index=True)
            continue

        # Check position consistency.
        error = False
        pos = [float(p) for p in cts[0].ImagePositionPatient]
        for ct in cts:
            ct_pos = [float(p) for p in ct.ImagePositionPatient]
            if ct_pos[:2] != pos[:2]:
                error = True
        if error:
            msg = "Inconsistent CT slice 'ImagePositionPatient' (x/y) found for patient '{patient}'."
            _trim_patient(dataset, pat, msg)
            data = {
                'patient-id': pat,
                'reason': 'INCONSISTENT-POSITION'
            }
            df = df.append(data, ignore_index=True)
            continue

        # Check x/y spacing consistency.
        error = False
        spac = cts[0].PixelSpacing 
        for ct in cts:
            if ct.PixelSpacing != spac:
                error = True
        if error:
            msg = f"Inconsistent CT slice 'PixelSpacing' found for patient '{patient}'."
            _trim_patient(dataset, pat, msg)
            data = {
                'patient-id': pat,
                'reason': 'INCONSISTENT-PIXEL-SPACING'
            }
            df = df.append(data, ignore_index=True)
            continue

        # Check z spacing consistency.
        z_pos = [ct.ImagePositionPatient[2] for ct in cts]
        z_diff = np.diff(z_pos)
        z_diff = [round(d, 3) for d in z_diff]
        z_diff = np.unique(z_diff)
        if len(z_diff) != 1:
            msg = f"Inconsistence spacing of CT slice 'ImagePositionPatient' (z) for patient '{patient}'. Got '{z_diff}'."
            _trim_patient(dataset, pat, msg)
            data = {
                'patient-id': pat,
                'reason': 'INCONSISTENT-Z-SPACING'
            }
            df = df.append(data, ignore_index=True)
            continue

    # Save summary.
    df = df.astype(cols)
    path = os.path.join(dataset.path, 'hierarchy', 'trimmed', 'summary.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def _trim_patient(
    dataset: 'DICOMDataset',
    pat_id: types.PatientID,
    error_msg: str) -> None:

    # Move patient to error folder.
    path = os.path.join(dataset.path, 'hierarchy', 'data', pat_id)
    new_path = os.path.join(dataset.path, 'hierarchy', 'trimmed', 'data', pat_id)
    shutil.move(path, new_path)

    # Write error message.
    msg = f"Patient '{pat_id}' trimmed from hierarchy for dataset '{dataset}'.\nError: {error_msg}"
    path = os.path.join(new_path, 'error.log')
    with open(path, 'w') as f:
        f.write(msg)

    # Log error message.
    logging.error(msg)
