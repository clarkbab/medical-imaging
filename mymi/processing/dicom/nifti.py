import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import shutil
from time import time
from typing import *
from tqdm import tqdm

from mymi.datasets.dataset import CT_FROM_REGEXP
from mymi.datasets.dicom import DicomDataset
from mymi.datasets.nifti import NiftiDataset, create as create_nifti, exists as exists_nifti, recreate as recreate_nifti
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ..processing import write_flag

ERROR_COLS = {
    'error': str
}
ERROR_INDEX = [
    'dataset',
    'patient-id'
]

def convert_to_nifti(
    dataset: str,
    anonymise_patients: bool = False,
    anonymise_series: bool = True,
    anonymise_studies: bool = True,
    convert_ct: bool = True,
    convert_dose: bool = True,
    convert_mr: bool = True,
    dest_dataset: Optional[str] = None,
    filter_pats_by_landmarks: bool = False,
    filter_pats_by_regions: bool = False,
    landmark_ids: Optional[LandmarkIDs] = 'all',
    pat_ids: PatientIDs = 'all',
    recreate: bool = False,
    recreate_patients: bool = False,   # Setting to False allows us to append new patients without removing existing.
    recreate_ct: bool = False,         # Setting to False allows us to add new data to a patient without removing existing.
    recreate_dose: bool = False,
    recreate_landmarks: bool = False,
    recreate_regions: bool = False,
    region_ids: Optional[RegionIDs] = 'all',
    ) -> None:
    logging.arg_log('Converting DicomDataset to NiftiDataset', ('dataset', 'anonymise_patients', 'anonymise_studies', 'anonymise_studies', 'region_ids'), (dataset, anonymise_patients, anonymise_studies, anonymise_studies, region_ids))
    start = time()

    # Load all patients.
    dicom_set = DicomDataset(dataset)
    okwargs = dict(pat_ids=pat_ids)
    if filter_pats_by_landmarks and landmark_ids is not None: 
        okwargs['landmark_ids'] = landmark_ids
    if filter_pats_by_regions and region_ids is not None:
        okwargs['region_ids'] = region_ids
    resolved_pat_ids = dicom_set.list_patients(**okwargs)

    # Create NIFTI dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    if exists_nifti(dest_dataset):
        if recreate:
            nifti_set = recreate_nifti(dest_dataset)
        else:
            nifti_set = NiftiDataset(dest_dataset)
    else:
        nifti_set = create_nifti(dest_dataset)

    # Remove markers.
    files = os.listdir(nifti_set.path)
    for f in files:
        if f.startswith('__DICOM_CONVERSION_TIME_MINS_'):
            os.remove(os.path.join(nifti_set.path, f))

    # Check if index is open and therefore can't be overwritten.
    filepath = os.path.join(nifti_set.path, 'index.csv')
    if os.path.exists(filepath):
        try:
            open(filepath, 'a')
        except PermissionError:
            logging.error(f"Index file '{filepath}' is currently open and cannot be overwritten. Please close it before running conversion.")
            return

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

    # Create or load index.
    cols = {
        'dataset': str,
        'patient-id': str,
        'study-id': str,
        'series-id': str,
        'dicom-dataset': str,
        'dicom-patient-id': str,
        'dicom-study-id': str,
        'dicom-series-id': str,
    }
    filepath = os.path.join(nifti_set.path, 'index.csv')
    if recreate or recreate_patients or not os.path.exists(filepath):
        index = pd.DataFrame(columns=cols.keys())
    else:
        index = load_csv(filepath, map_types=cols)

        # Remove overwritten patients.
        for p in resolved_pat_ids:
            index = index[index['dicom-patient-id'] != p] 

    # Determine IDs.
    if anonymise_patients:
        if recreate or recreate_patients or len(index) == 0:
            start_idx = 0
        else:
            existing_pat_ids = index['patient-id'].unique()
            start_idx = list(sorted(int(i.split('_')[1]) for i in existing_pat_ids))[-1] + 1
            logging.warning(f"Existing dataset has {len(existing_pat_ids)} patients, adding new patients starting from {start_idx}.")
        anon_pat_ids = [f'pat_{start_idx + i}' for i in range(len(resolved_pat_ids))]
    else:
        anon_pat_ids = resolved_pat_ids

    # Remove existing patient data.
    if not recreate:
        if recreate_patients:
            filepath = os.path.join(nifti_set.path, 'data', 'patients')
            shutil.rmtree(filepath)
        else:
            # Remove series for any patients we're overwriting.
            for ap in anon_pat_ids:
                pat_dirpath = os.path.join(nifti_set.path, 'data', 'patients', ap)
                if os.path.exists(pat_dirpath):
                    study_ids = os.listdir(pat_dirpath)
                    for s in study_ids:
                        recreate_mods = [recreate_ct, recreate_dose, recreate_landmarks, recreate_regions]
                        mods = ['ct', 'dose', 'landmarks', 'regions']
                        for r, m in zip(recreate_mods, mods):
                            dirpath = os.path.join(pat_dirpath, s, m)
                            if r and os.path.exists(dirpath):
                                shutil.rmtree(dirpath)

    # Write all patient data.
    for p, ap in tqdm(zip(resolved_pat_ids, anon_pat_ids), total=len(resolved_pat_ids)):
        pat = dicom_set.patient(p)
                
        study_ids = pat.list_studies()
        anon_study_id = 0
        for s in study_ids:
            study = pat.study(s)
            if not study.has_ct:
                continue
            
            # Get Nifti study ID.
            if anonymise_studies:
                nifti_study_id = f'study_{anon_study_id}'
                anon_study_id += 1
            else:
                nifti_study_id = s

            anon_series_id = 0
            if ct_from is None:
                # Convert CT series.
                ct_series_ids = study.list_series('ct')
                for sr in ct_series_ids:
                    series = study.series(sr, 'ct')
                    
                    # Get Nifti series ID.
                    if anonymise_series:
                        nifti_series_id = f'series_{anon_series_id}'
                        anon_series_id += 1
                    else:
                        nifti_series_id = sr
                    
                    # Create Nifti CT.
                    # Doesn't overwrite data, if we want to replace some existing data, need to use
                    # a 'recreate' tag, which will remove existing patient data.
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study_id, 'ct', f'{nifti_series_id}.nii.gz')
                    if convert_ct and not os.path.exists(filepath):
                        save_nifti(series.data, filepath, spacing=series.spacing, offset=series.offset)

                    # Add index entry.
                    data = {
                        'dataset': dest_dataset,
                        'patient-id': ap,
                        'study-id': nifti_study_id,
                        'series-id': nifti_series_id,
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                    }
                    index = append_row(index, data)

                # Convert MR series.
                mr_series_ids = study.list_series('mr')
                for sr in mr_series_ids:
                    series = study.series(sr, 'mr')
                    
                    # Get Nifti series ID.
                    if anonymise_series:
                        nifti_series_id = f'series_{anon_series_id}'
                        anon_series_id += 1
                    else:
                        nifti_series_id = sr
                    
                    # Create Nifti MR.
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study_id, 'mr', f'{nifti_series_id}.nii.gz')
                    if convert_mr and not os.path.exists(filepath):
                        save_nifti(series.data, filepath, spacing=series.spacing, offset=series.offset)

                    # Add index entry.
                    data = {
                        'dataset': dataset,
                        'patient-id': ap,
                        'study-id': nifti_study_id,
                        'series-id': nifti_series_id,
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                    }
                    index = append_row(index, data)

            # Convert RTSTRUCT series.
            rtstruct_series_ids = study.list_series('rtstruct')
            for sr in rtstruct_series_ids:
                series = study.series(sr, 'rtstruct')

                # Get Nifti series ID.
                if anonymise_series:
                    nifti_series_id = f'series_{anon_series_id}'
                    anon_series_id += 1
                else:
                    nifti_series_id = sr

                # Create region NIFTIs.
                if region_ids is not None:
                    ref_ct = series.ref_ct
                    region_data = series.region_data(region_ids=region_ids, regions_ignore_missing=True)
                    for r, data in region_data.items():
                        filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study_id, 'regions', nifti_series_id, f'{r}.nii.gz')
                        if not os.path.exists(filepath):
                            save_nifti(data, filepath, spacing=ref_ct.spacing, offset=ref_ct.offset)

                # Create landmarks.
                if landmark_ids is not None:
                    lm_df = series.landmark_data(landmark_ids=landmark_ids, show_ids=False)
                    if lm_df is not None:
                        filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study_id, 'landmarks', f'{nifti_series_id}.csv')
                        if not os.path.exists(filepath):
                            save_csv(lm_df, filepath)

            # Convert RTDOSE series.
            rtdose_series_ids = study.list_series('rtdose')
            for sr in rtdose_series_ids:
                rtdose_series = study.series(sr, 'rtdose')

                # Get Nifti series ID.
                if anonymise_series:
                    nifti_series_id = f'series_{anon_series_id}'
                    anon_series_id += 1
                else:
                    nifti_series_id = sr

                # Create RTDOSE NIFTI.
                filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study_id, 'dose', f'{nifti_series_id}.nii.gz')
                if convert_dose and not os.path.exists(filepath):
                    save_nifti(rtdose_series.data, filepath, spacing=rtdose_series.spacing, offset=rtdose_series.offset)

    # Save index.
    if len(index) > 0:
        index = index.astype(cols)
        index = index.sort_values(['patient-id', 'study-id', 'series-id'])   # Required if adding patients to existing converted dataset.
    filepath = os.path.join(nifti_set.path, 'index.csv')
    save_csv(index, filepath)

    # Save indexing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    filepath = os.path.join(nifti_set.path, f'__DICOM_CONVERSION_TIME_MINS_{mins}__')
    Path(filepath).touch()

def convert_to_nifti_replan(
    dataset: str,
    dicom_dataset: Optional[str] = None,
    region: Regions = 'all',
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
    study_df = pd.read_csv(filepath, dtype={ 'patient-id': str })
    pat_ids = list(sorted(np.unique(study_df['patient-id'])))

    if anonymise:
        cols = {
            'patient-id': str,
            'origin-dataset': str,
            'origin-patient-id': str,
            'origin-study-id': str
        }
        df = pd.DataFrame(columns=cols.keys())

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
            region_data = study.region_data(regions=regions, regions_ignore_missing=True)
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
