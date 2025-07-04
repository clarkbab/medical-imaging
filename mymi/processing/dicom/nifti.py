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

from mymi.datasets.shared import CT_FROM_REGEXP
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
    anonymise_patients: bool = True,
    anonymise_series: bool = True,
    anonymise_studies: bool = True,
    dest_dataset: Optional[str] = None,
    recreate: bool = False,
    regions: Optional[Regions] = 'all') -> None:
    logging.arg_log('Converting DicomDataset to NiftiDataset', ('dataset', 'anonymise_patients', 'anonymise_studies', 'anonymise_studies', 'regions'), (dataset, anonymise_patients, anonymise_studies, anonymise_studies, regions))
    start = time()

    # Create NIFTI dataset.
    dicom_set = DicomDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': dicom_set.list_regions })
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    if exists_nifti(dest_dataset):
        if recreate:
            nifti_set = recreate_nifti(dest_dataset)
        else:
            nifti_set = NiftiDataset(dest_dataset)
            # Remove 'patients' data to ensure there are no leftovers. Preserve other data, e.g. predictions, evaluations, etc.
            filepath = os.path.join(nifti_set.path, 'data', 'patients')
            if os.path.exists(filepath):
                shutil.rmtree(filepath)
    else:
        nifti_set = create_nifti(dest_dataset)

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
    pat_ids = dicom_set.list_patients(regions=regions)

    # Create mapping index.
    cols = {
        'dicom-dataset': str,
        'dicom-patient-id': str,
        'dicom-study-id': str,
        'dicom-series-id': str,
        'nifti-dataset': str,
        'nifti-patient-id': str,
        'nifti-study-id': str,
        'nifti-series-id': str,
    }
    index_df = pd.DataFrame(columns=cols.keys())

    anon_patient_id = 0
    for p in tqdm(pat_ids):
        pat = dicom_set.patient(p)
        
        # Get Nifti patient ID.
        if anonymise_patients:
            nifti_pat_id = f'pat_{anon_patient_id}'
            anon_patient_id += 1
        else:
            nifti_pat_id = p
                
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
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', nifti_pat_id, nifti_study_id, 'ct', f'{nifti_series_id}.nii.gz')
                    save_nifti(series.data, filepath, spacing=series.spacing, offset=series.offset)

                    # Add index entry.
                    data = {
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                        'nifti-dataset': dataset,
                        'nifti-patient-id': nifti_pat_id,
                        'nifti-study-id': nifti_study_id,
                        'nifti-series-id': nifti_series_id,
                    }
                    index_df = append_row(index_df, data)

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
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', nifti_pat_id, nifti_study_id, 'mr', f'{nifti_series_id}.nii.gz')
                    save_nifti(series.data, filepath, spacing=series.spacing, offset=series.offset)

                    # Add index entry.
                    data = {
                        'dicom-dataset': dataset,
                        'dicom-patient-id': p,
                        'dicom-study-id': s,
                        'dicom-series-id': sr,
                        'nifti-dataset': dataset,
                        'nifti-patient-id': nifti_pat_id,
                        'nifti-study-id': nifti_study_id,
                        'nifti-series-id': nifti_series_id,
                    }
                    index_df = append_row(index_df, data)

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
                ref_ct = series.ref_ct
                region_data = series.regions_data(regions=regions, regions_ignore_missing=True)
                for r, data in region_data.items():
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', nifti_pat_id, nifti_study_id, 'regions', nifti_series_id, f'{r}.nii.gz')
                    save_nifti(data, filepath, spacing=ref_ct.spacing, offset=ref_ct.offset)

                # Create landmarks.
                lm_df = series.landmarks_data()
                if lm_df is not None:
                    lm_df['patient-id'] = nifti_pat_id
                    lm_df['study-id'] = nifti_study_id
                    filepath = os.path.join(nifti_set.path, 'data', 'patients', nifti_pat_id, nifti_study_id, 'landmarks', f'{nifti_series_id}.csv')
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
                filepath = os.path.join(nifti_set.path, 'data', 'patients', nifti_pat_id, nifti_study_id, 'dose', f'{nifti_series_id}.nii.gz')
                save_nifti(rtdose_series.data, filepath, spacing=rtdose_series.spacing, offset=rtdose_series.offset)

    # Save index.
    if len(index_df) > 0:
        index_df = index_df.astype(cols)
    filepath = os.path.join(nifti_set.path, 'index.csv')
    save_csv(index_df, filepath)

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
            region_data = study.regions_data(regions=regions, regions_ignore_missing=True)
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
