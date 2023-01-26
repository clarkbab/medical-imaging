import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
from typing import Optional
from tqdm import tqdm

from mymi.dataset.dicom import DICOMDataset
from mymi.dataset.nifti import recreate as recreate_nifti
from mymi import logging
from mymi import types
from mymi.utils import save_csv

from .dataset import write_flag

def convert_to_nifti(
    dataset: 'Dataset',
    regions: types.PatientRegions = 'all',
    anonymise: bool = False) -> None:
    # Create NIFTI dataset.
    nifti_ds = recreate_nifti(dataset.name)

    logging.info(f"Converting dataset '{dataset}' to dataset '{nifti_ds}', with regions '{regions}' and anonymise '{anonymise}'.")

    # Load all patients.
    pats = dataset.list_patients(regions=regions)

    if anonymise:
        # Create CT map. Index of map will be the anonymous ID.
        map_df = pd.DataFrame(pats, columns=['patient-id']).reset_index().rename(columns={ 'index': 'anon-id' })

        # Save map.
        save_csv(map_df, 'anon-maps', f'{dataset.name}.csv', overwrite=True)

    for pat in tqdm(pats):
        # Get anonymous ID.
        if anonymise:
            anon_id = map_df[map_df['patient-id'] == pat].index.values[0]
            filename = f'{anon_id}.nii.gz'
        else:
            filename = f'{pat}.nii.gz'

        # Create CT NIFTI.
        patient = dataset.patient(pat)
        data = patient.ct_data
        spacing = patient.ct_spacing
        offset = patient.ct_offset
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(nifti_ds.path, 'data', 'ct', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        pat_regions = patient.list_regions(whitelist=regions)
        region_data = patient.region_data(regions=pat_regions)
        for region, data in region_data.items():
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(nifti_ds.path, 'data', 'regions', region, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

        # Create RTDOSE NIFTI.
        try:
            patient = dataset.patient(pat, load_default_rtdose=True) 
            dose_data = patient.dose_data
            img = Nifti1Image(dose_data, affine)
            filepath = os.path.join(nifti_ds.path, 'data', 'dose', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)
        except ValueError as e:
            logging.error(str(e))

    # Indicate success.
    write_flag(nifti_ds, '__CONVERT_FROM_NIFTI_END__')

def convert_to_nifti_multiple_studies(
    dataset: 'Dataset',
    regions: types.PatientRegions = 'all',
    anonymise: bool = False) -> None:
    # Create NIFTI dataset.
    nifti_ds = recreate_nifti(dataset)
    logging.arg_log('Converting dataset to NIFTI', ('dataset', 'regions', 'anonymise'), (dataset, regions, anonymise))

    # Get all patients.
    set = DICOMDataset(dataset)
    filepath = os.path.join(set.path, 'patient-studies.csv')
    if not os.path.exists(filepath):
        raise ValueError(f"File '<dataset>/patient-studies.csv' not found.")
    study_df = pd.read_csv(filepath)
    pat_ids = list(sorted(np.unique(study_df['patient-id'])))

    if anonymise:
        # Create ID mapping to anonymise patient IDs.
        map_df = pd.DataFrame(pat_ids, columns=['patient-id']).reset_index().rename(columns={ 'index': 'anon-id' })
        filepath = os.path.join(set.path, 'nifti-map.csv')
        map_df.to_csv(filepath, index=False)

    for pat_id in tqdm(pat_ids[:5]):
        # Get anonymous ID.
        if anonymise:
            nifti_id = map_df[map_df['patient-id'] == pat_id].iloc[0]['anon-id']
        else:
            nifti_id = pat_id

        # Get study IDs.
        study_ids = study_df[study_df['patient-id'] == pat_id]['study-id'].values

        for i, study_id in enumerate(study_ids):
            # Create CT nifti for study.
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
            filepath = os.path.join(nifti_ds.path, 'data', 'ct', f'{nifti_id}-{i}.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

            # Create region niftis for study.
            pat_regions = study.list_regions()
            pat_regions = [r for r in pat_regions if r in regions]
            region_data = study.region_data(regions=pat_regions)
            for region, data in region_data.items():
                img = Nifti1Image(data.astype(np.int32), affine)
                filepath = os.path.join(nifti_ds.path, 'data', 'regions', region, f'{nifti_id}-{i}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

            # Create RTDOSE niftis for study.
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
                filepath = os.path.join(nifti_ds.path, 'data', 'dose', '{nifti_id}-{i}.nii.gz')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                nib.save(img, filepath)

    # Indicate success.
    write_flag(nifti_ds, '__CONVERT_FROM_NIFTI_END__')
