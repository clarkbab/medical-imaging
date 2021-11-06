from distutils.dir_util import copy_tree
import numpy as np
import pandas as pd
import pydicom as dcm
import os
import shutil
from tqdm import tqdm
from typing import Callable

from mymi import logging
from mymi import types

from .dicom_series import DICOMModality

def require_hierarchy(fn: Callable) -> Callable:
    def _require_hierarchy_wrapper(
        dataset: 'DICOMDataset',
        *args,
        **kwargs):
        if not _hierarchy_exists(dataset):
            _build_hierarchy(dataset)
            _trim_hierarchy(dataset)
        return fn(dataset, *args, **kwargs)
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
def _build_hierarchy(dataset: 'DICOMDataset') -> None:
    logging.info(f"Building hierarchy for dataset '{dataset}'...")

    # Load all dicom files.
    raw_path = os.path.join(dataset.path, 'raw')
    if not os.path.exists(raw_path):
        raise ValueError(f"No 'raw' folder found for dataset '{dataset}'.")

    dicom_files = []
    for root, _, files in tqdm(os.walk(raw_path)):
        for f in files:
            # Check if DICOM file.
            filepath = os.path.join(root, f)
            try:
                dicom = dcm.read_file(filepath, stop_before_pixels=True)
            except dcm.errors.InvalidDicomError:
                continue

            # Get patient ID.
            pat_id = dicom.PatientID

            # Get modality.
            mod = dicom.Modality.lower()
            if not mod in ('ct', 'rtstruct'):
                continue

            # Get series UID.
            series_UID = dicom.SeriesInstanceUID

            # Get study UID.
            study_UID = dicom.StudyInstanceUID

            # Add '.dcm' extension.
            filename = os.path.basename(f)
            if not filename.endswith('.dcm'):
                filename = f'{filename}.dcm'

            # Copy DICOM file.
            newpath = os.path.join(dataset.path, 'hierarchy', 'data', pat_id, study_UID, mod, series_UID, filename)
            os.makedirs(os.path.dirname(newpath), exist_ok=True)
            shutil.copy(filepath, newpath)

@_rollback_hierarchy
def _trim_hierarchy(dataset: 'DICOMDataset') -> None:
    logging.info(f"Trimming hierarchy for dataset '{dataset}'...")

    # Create summary.
    cols = {
        'patient-id': str,
        'study-id': str,
        'series-id': str,
        'error-desc': str,
        'error-message': str
    }
    error_df = pd.DataFrame(columns=cols.keys())

    # Get patients.
    pats = dataset.list_patients()

    for pat_id in tqdm(pats):
        patient = dataset.patient(pat_id, load_default_series=False)
        studies = patient.list_studies()

        for study_id in studies:
            study = patient.study(study_id)
            ct_series_ids = study.list_series('ct')
            rt_series_ids = study.list_series('rtstruct')

            for ct_id in ct_series_ids:
                ct_series = study.series(ct_id, 'ct')
                cts = ct_series.get_cts()

                # Series-level checks.

                # CHECK: CT series has no missing slices.
                cts = list(sorted(cts, key=lambda c: c.InstanceNumber))
                nums = [int(c.InstanceNumber) for c in cts]
                nums_diff = np.unique(np.diff(nums))
                if len(nums_diff) != 1 or nums_diff[0] != 1:
                    error_code = 'CT-MISSING-SLICES'
                    error_message = f"Missing slices (non-contigous 'InstanceNumber') for CT series '{ct_series}'."
                    error_df = _trim_series(ct_series, error_df, error_code, error_message)
                    continue

                # CHECK: CT series slices have consistent orientation.
                error = False
                ori = cts[0].ImageOrientationPatient
                for ct in cts:
                    if ct.ImageOrientationPatient != ori:
                        error = True
                if error:
                    error_code = 'CT-INCONSISTENT-ORIENTATION'
                    error_message = f"Inconsistent orientation ('ImageOrientationPatient') for CT series '{ct_series}'."
                    error_df = _trim_series(ct_series, error_df, error_code, error_message)
                    continue

                # CHECK: CT series slices have standard orientation.
                if ori != [1, 0, 0, 0, 1, 0]:
                    error_code = 'CT-NON-STANDARD-ORIENTATION'
                    error_message = f"Non-standard orientation ('ImageOrientationPatient') for CT series '{ct_series}'."
                    error_df = _trim_series(ct_series, error_df, error_code, error_message)
                    continue

                # CHECK: CT series slices have consistent x/y position.
                error = False
                pos = [float(p) for p in cts[0].ImagePositionPatient]
                for ct in cts:
                    ct_pos = [float(p) for p in ct.ImagePositionPatient]
                    if ct_pos[:2] != pos[:2]:
                        error = True
                if error:
                    error_code = 'CT-INCONSISTENT-XY-POSITION'
                    error_message = f"Inconsistent x/y position ('ImagePositionPatient') for CT series '{ct_series}'."
                    error_df = _trim_series(ct_series, error_df, error_code, error_message)
                    continue

                # CHECK: CT series slices have consistent x/y spacing.
                error = False
                spacing = cts[0].PixelSpacing 
                for ct in cts:
                    if ct.PixelSpacing != spacing:
                        error = True
                if error:
                    error_code = 'CT-INCONSISTENT-XY-SPACING'
                    error_message = f"Inconsistent x/y spacing ('PixelSpacing') for CT series '{ct_series}'."
                    error_df = _trim_series(ct_series, error_df, error_code, error_message)
                    continue

                # CHECK: CT series slices have consistent z spacing (to 3 d.p).
                z_pos = [ct.ImagePositionPatient[2] for ct in cts]
                z_diff = np.diff(z_pos)
                z_diff = [round(d, 3) for d in z_diff]
                z_diff = np.unique(z_diff)
                if len(z_diff) != 1:
                    error_code = 'CT-INCONSISTENT-Z-SPACING'
                    error_message = f"Inconsistent spacing of CT slice 'ImagePositionPatient' (z) for patient '{patient}'. Got '{z_diff}'."
                    error_df = _trim_series(ct_series, error_df, error_code, error_message)
                    continue

            # Reload CT series after trimming invalid series.
            valid_ct_series_ids = study.list_series('ct')

            for rt_id in rt_series_ids:
                rt_series = study.series(rt_id, 'rtstruct', load_ref_ct=False)
                rt = rt_series.get_rtstruct()

                # CHECK: RTSTRUCT series has single file.
                if len(os.listdir(rt_series.path)) != 1:
                    error_code = 'RTSTRUCT-MULTIPLE-FILES'
                    error_message = f"Multiple files found for RTSTRUCT series '{rt_series}'."
                    error_df = _trim_series(rt_series, error_df, error_code, error_message)
                    continue

                # CHECK: RTSTRUCT series references valid CT series.
                ct_id = rt.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                if not ct_id in valid_ct_series_ids:
                    error_code = 'RTSTRUCT-NO-CT'
                    error_message = f"No valid CT series found for RTSTRUCT series '{rt_series}'."
                    error_df = _trim_series(rt_series, error_df, error_code, error_message)
                    continue

            # Study-level checks.

            # CHECK: Study has valid RTSTRUCT series.
            if len(study.list_series('rtstruct')) == 0:
                error_code = 'STUDY-NO-RTSTRUCT'
                error_message = f"No valid RTSTRUCT series found for study '{study}'."
                error_df = _trim_study(study, error_df, error_code, error_message)
                continue
        
        # Patient-level checks.

        # CHECK: Patient has a valid study.
        if len(patient.list_studies()) == 0:
            error_code = 'PATIENT-NO-STUDY'
            error_message = f"No valid study found for patient '{patient}'."
            error_df = _trim_patient(patient, error_df, error_code, error_message)
            continue

    # Save summary.
    error_df = error_df.astype(cols)
    path = os.path.join(dataset.path, 'hierarchy', 'trimmed', 'errors.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    error_df.to_csv(path, index=False)

def _trim_patient(
    patient: 'DICOMPatient',
    error_df: pd.DataFrame,
    error_desc: str,
    error_message: str) -> pd.DataFrame:
    # Move patient to trimmed folder if doesn't already exist.
    folders = patient.path.split(os.path.sep)
    folders.insert(-2, 'trimmed')
    trim_path = os.path.sep.join(folders)
    _merge_copy(patient.path, trim_path)

    # Log error.
    data = {
        'patient-id': patient.id,
        'error-desc': error_desc,
        'error-message': error_message
    }
    error_df = error_df.append(data, ignore_index=True)
    return error_df

def _trim_study(
    study: 'DICOMStudy',
    error_df: pd.DataFrame,
    error_desc: str,
    error_message: str) -> pd.DataFrame:
    # Move study to trimmed folder if doesn't already exist.
    folders = study.path.split(os.path.sep)
    folders.insert(-3, 'trimmed')
    trim_path = os.path.sep.join(folders)
    _merge_copy(study.path, trim_path)

    # Log error.
    data = {
        'patient-id': study.patient.id,
        'study-id': study.id,
        'error-desc': error_desc,
        'error-message': error_message
    }
    error_df = error_df.append(data, ignore_index=True)
    return error_df

def _trim_series(
    series: 'DICOMSeries',
    error_df: pd.DataFrame,
    error_desc: str,
    error_message: str) -> pd.DataFrame:
    # Move series to trimmed folder.
    folders = series.path.split(os.path.sep)
    folders.insert(-5, 'trimmed')
    trim_path = os.path.sep.join(folders)
    _merge_copy(series.path, trim_path)

# Log error.
    data = {
        'patient-id': series.study.patient.id,
        'study-id': series.study.id,
        'series-id': series.id, 
        'error-desc': error_desc,
        'error-message': error_message
    }
    error_df = error_df.append(data, ignore_index=True)
    return error_df

def _merge_copy(
    source: str,
    dest: str) -> None:
    copy_tree(source, dest)
    shutil.rmtree(source)
