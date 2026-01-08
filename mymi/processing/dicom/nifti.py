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
from mymi.datasets.dicom import DicomDataset, DicomStudy
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
    dry_run: bool = True,
    filter_pats_by_landmarks: bool = False,
    filter_pats_by_regions: bool = False,
    group: Optional[PatientGroups] = 'all',
    landmark: Optional[LandmarkIDs] = 'all',
    pat: PatientIDs = 'all',
    recreate: bool = False,
    recreate_patient: bool = False,   # Setting to False allows us to append new patients without removing existing.
    recreate_ct: bool = False,         # Setting to False allows us to add new data to a patient without removing existing.
    recreate_dose: bool = False,
    recreate_landmarks: bool = False,
    recreate_regions: bool = False,
    region: Optional[RegionIDs] = 'all',
    study_sort: Optional[Callable[DicomStudy, int]] = None,
    ) -> None:
    logging.arg_log('Converting DicomDataset to NiftiDataset', ('dataset', 'anonymise_patients', 'anonymise_studies', 'anonymise_series', 'region'), (dataset, anonymise_patients, anonymise_studies, anonymise_studies, region))
    start = time()

    # Load all patients.
    dicom_set = DicomDataset(dataset)
    okwargs = dict(group=group, pat=pat)
    if filter_pats_by_landmarks and landmark is not None: 
        okwargs['landmark'] = landmark
    if filter_pats_by_regions and region is not None:
        okwargs['region'] = region
    resolved_pat_ids = dicom_set.list_patients(**okwargs)

    # Create NIFTI dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    if exists_nifti(dest_dataset):
        if recreate:
            nifti_set = recreate_nifti(dest_dataset, dry_run=dry_run)
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

    # Copy 'groups.csv' file.
    filepath = os.path.join(dicom_set.path, 'groups.csv')
    if os.path.exists(filepath):
        destpath = os.path.join(nifti_set.path, 'groups.csv')
        shutil.copy(filepath, destpath)

    # Create or load index.
    # Each row of the index refers to a specific series.
    cols = {
        'dataset': str,
        'patient-id': str,
        'study-id': str,
        'series-id': str,
        'modality': str,
        'dicom-dataset': str,
        'dicom-patient-id': str,
        'dicom-study-id': str,
        'dicom-series-id': str,
        'dicom-modality': str,
    }
    filepath = os.path.join(nifti_set.path, 'index.csv')
    if recreate or not os.path.exists(filepath):
        index = pd.DataFrame(columns=cols.keys())
    else:
        index = load_csv(filepath, map_types=cols)

        # Remove overwritten patients.
        for p in resolved_pat_ids:
            index = index[index['dicom-patient-id'] != p] 

    # Determine IDs.
    if anonymise_patients:
        if recreate or len(index) == 0:
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
        # Remove series for any patients we're overwriting.
        for ap in anon_pat_ids:
            pat_dirpath = os.path.join(nifti_set.path, 'data', 'patients', ap)
            if os.path.exists(pat_dirpath):
                if recreate_patient:
                    shutil.rmtree(pat_dirpath)
                else:
                    studies = os.listdir(pat_dirpath)
                    for s in studies:
                        recreate_mods = [recreate_ct, recreate_dose, recreate_landmarks, recreate_regions]
                        mods = ['ct', 'dose', 'landmarks', 'regions']
                        for r, m in zip(recreate_mods, mods):
                            dirpath = os.path.join(pat_dirpath, s, m)
                            if r and os.path.exists(dirpath):
                                shutil.rmtree(dirpath)

    # Write all patient data.
    for p, ap in tqdm(zip(resolved_pat_ids, anon_pat_ids), total=len(resolved_pat_ids)):
        pat = dicom_set.patient(p)
        logging.info(pat)
                
        studys = pat.list_studies(sort=study_sort)
        anon_study = 0
        for s in studys:
            study = pat.study(s)
            logging.info(study)

            if not study.has_ct:
                logging.warning(f"Skipping study {study} due to no CT series. CT is required for RTSTRUCT reference geometry.")
                continue
            
            # Get Nifti study ID.
            if anonymise_studies:
                nifti_study = f'study_{anon_study}'
                anon_study += 1
            else:
                nifti_study = s

            anon_series = 0
            if ct_from is None:
                # Convert CT series.
                ct_serieses = study.list_series('ct')
                for sr in ct_serieses:
                    series = study.series(sr, 'ct')
                    logging.info(series)
                    
                    # Get Nifti series ID.
                    if anonymise_series:
                        nifti_series = f'series_{anon_series}'
                        anon_series += 1
                    else:
                        nifti_series = sr
                    
                    # Create Nifti CT.
                    # Doesn't overwrite data, if we want to replace some existing data, need to use
                    # a 'recreate' tag, which will remove existing patient data.
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study, 'ct', f'{nifti_series}.nii.gz')
                    if convert_ct and not os.path.exists(filepath):
                        save_nifti(series.data, filepath, spacing=series.spacing, origin=series.origin)

                    # Add index entry.
                    data = {
                        'dataset': dest_dataset,
                        'patient-id': ap,
                        'study-id': nifti_study,
                        'series-id': nifti_series,
                        'modality': 'ct',
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                        'dicom-modality': 'ct',
                    }
                    index = append_row(index, data)

                # Convert MR series.
                mr_serieses = study.list_series('mr')
                for sr in mr_serieses:
                    series = study.series(sr, 'mr')
                    logging.info(series)
                    
                    # Get Nifti series ID.
                    if anonymise_series:
                        nifti_series = f'series_{anon_series}'
                        anon_series += 1
                    else:
                        nifti_series = sr
                    
                    # Create Nifti MR.
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study, 'mr', f'{nifti_series}.nii.gz')
                    if convert_mr and not os.path.exists(filepath):
                        save_nifti(series.data, filepath, spacing=series.spacing, origin=series.origin)

                    # Add index entry.
                    data = {
                        'dataset': dataset,
                        'patient-id': ap,
                        'study-id': nifti_study,
                        'series-id': nifti_series,
                        'modality': 'mr',
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                        'dicom-modality': 'mr',
                    }
                    index = append_row(index, data)

            # Convert RTSTRUCT series.
            rtstruct_serieses = study.list_series('rtstruct')
            for sr in rtstruct_serieses:
                series = study.series(sr, 'rtstruct')
                logging.info(series)

                # Get Nifti series ID.
                if anonymise_series:
                    nifti_series = f'series_{anon_series}'
                    anon_series += 1
                else:
                    nifti_series = sr

                # Create region NIFTIs.
                if region is not None:
                    ref_ct = series.ref_ct
                    print(series)
                    print(region)
                    regions_data = series.regions_data(region=region, regions_ignore_missing=True)
                    for r, data in regions_data.items():
                        filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study, 'regions', nifti_series, f'{r}.nii.gz')
                        if not os.path.exists(filepath):
                            save_nifti(data, filepath, spacing=ref_ct.spacing, origin=ref_ct.origin)

                    # Add index entry.
                    data = {
                        'dataset': dataset,
                        'patient-id': ap,
                        'study-id': nifti_study,
                        'series-id': nifti_series,
                        'modality': 'regions',
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                        'dicom-modality': 'rtstruct',
                    }
                    index = append_row(index, data)

                # Create landmarks.
                if landmark is not None:
                    lm_df = series.landmarks_data(landmark=landmark, show_ids=False)
                    if lm_df is not None:
                        filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study, 'landmarks', f'{nifti_series}.csv')
                        if not os.path.exists(filepath):
                            save_csv(lm_df, filepath)

                        # Add index entry.
                        data = {
                            'dataset': dataset,
                            'patient-id': ap,
                            'study-id': nifti_study,
                            'series-id': nifti_series,
                            'modality': 'landmarks',
                            'dicom-dataset': dataset,
                            'dicom-patient-id': p,
                            'dicom-study-id': s,
                            'dicom-series-id': sr,
                            'dicom-modality': 'rtstruct',
                        }
                        index = append_row(index, data)

            # Convert RTDOSE series.
            rtdose_serieses = study.list_series('rtdose')
            for sr in rtdose_serieses:
                series = study.series(sr, 'rtdose')
                logging.info(series)

                # Get Nifti series ID.
                if anonymise_series:
                    nifti_series = f'series_{anon_series}'
                    anon_series += 1
                else:
                    nifti_series = sr

                # Create RTDOSE NIFTI.
                filepath = os.path.join(nifti_set.path, 'data', 'patients', ap, nifti_study, 'dose', f'{nifti_series}.nii.gz')
                if convert_dose and not os.path.exists(filepath):
                    save_nifti(series.data, filepath, spacing=series.spacing, origin=series.origin)

                # Add index entry.
                data = {
                    'dataset': dataset,
                    'patient-id': ap,
                    'study-id': nifti_study,
                    'series-id': nifti_series,
                    'modality': 'dose',
                    'dicom-dataset': dataset,
                    'dicom-patient-id': p,
                    'dicom-study-id': s,
                    'dicom-series-id': sr,
                    'dicom-modality': 'rtdose',
                }
                index = append_row(index, data)

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
        studys = study_df[study_df['patient-id'] == pat_id]['study-id'].values

        for j, study in enumerate(studys):
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
                    'origin-study-id': study,
                }
                df = append_row(df, data)

            # Create CT NIFTI for study.
            pat = set.patient(pat_id)
            study = pat.study(study)
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            ct_origin = study.ct_origin
            affine = np.array([
                [ct_spacing[0], 0, 0, ct_origin[0]],
                [0, ct_spacing[1], 0, ct_origin[1]],
                [0, 0, ct_spacing[2], ct_origin[2]],
                [0, 0, 0, 1]])
            img = Nifti1Image(ct_data, affine)
            filepath = os.path.join(nifti_set.path, 'data', 'ct', f'{nifti_id}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

            # Create region NIFTIs for study.
            regions_data = study.regions_data(regions=regions, regions_ignore_missing=True)
            for region, data in regions_data.items():
                img = Nifti1Image(data.astype(np.int32), affine)
                filepath = os.path.join(nifti_set.path, 'data', 'regions', region, f'{nifti_id}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

            # Create RTDOSE NIFTIs for study.
            dose_data = study.dose_data
            if dose_data is not None:
                dose_spacing = study.dose_spacing
                dose_origin = study.dose_origin
                affine = np.array([
                    [dose_spacing[0], 0, 0, dose_origin[0]],
                    [0, dose_spacing[1], 0, dose_origin[1]],
                    [0, 0, dose_spacing[2], dose_origin[2]],
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
