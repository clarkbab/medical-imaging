from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
from typing import Dict, Optional, Union

from mymi import config
from mymi.dataset import NIFTIDataset
from mymi import logging
from mymi.transforms import register_image, register_label, resample_3D
from mymi.types import PatientID, PatientRegions
from mymi.utils import arg_to_list

def create_patient_registration(
    dataset: str,
    fixed_pat_id: PatientID,
    moving_pat_id: PatientID) -> None:
    logging.info(f"Registering patient '{moving_pat_id}' CT to patient '{fixed_pat_id}' CT.")

    # Load CT data.
    set = NIFTIDataset(dataset)
    fixed_pat = set.patient(fixed_pat_id)
    fixed_ct = fixed_pat.ct_data
    fixed_spacing = fixed_pat.ct_spacing
    fixed_offset = fixed_pat.ct_offset
    moving_pat = set.patient(moving_pat_id)
    moving_ct = moving_pat.ct_data
    moving_spacing = moving_pat.ct_spacing

    # Resample moving image to fixed image spacing to aid registration.
    moving_ct = resample_3D(moving_ct, spacing=moving_spacing, output_spacing=fixed_spacing)

    # Perform CT registration.
    reg_ct, reg_transform = register_image(fixed_ct, moving_ct, fixed_spacing, fixed_spacing, return_transform=True)

    # Save registered CT.
    filepath = os.path.join(set.path, 'data', 'registrations', 'ct', f'{moving_pat_id}-{fixed_pat_id}.nii.gz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=reg_ct)

    # Load region data.
    moving_region_data = moving_pat.region_data()
    logging.info(f"Registering patient '{moving_pat_id}' regions ({list(moving_region_data.keys())}) to patient '{fixed_pat_id}'.")
    for region, moving_label in moving_region_data.items():
        # Register label to fixed image spacing.
        moving_label = resample_3D(moving_label, spacing=moving_spacing, output_spacing=fixed_spacing)

        # Apply CT registration transform.
        reg_label = register_label(moving_label, fixed_spacing, fixed_spacing, fixed_ct.shape, reg_transform)

        # Save registered region.
        filepath = os.path.join(config.directories.registrations, 'data', dataset, f'{moving_pat_id}-{fixed_pat_id}', f'{region}.npz')
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        np.savez_compressed(filepath, data=reg_label)

def load_patient_registration(
    dataset: str,
    fixed_pat_id: PatientID,
    moving_pat_id: PatientID,
    region: Optional[PatientRegions] = 'all',
    region_ignore_missing: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    # Load CT registration.
    filepath = os.path.join(config.directories.registrations, 'data', dataset, f'{moving_pat_id}-{fixed_pat_id}', 'ct.npz') 
    ct_data = np.load(filepath)['data']

    # Load region registrations.
    moving_pat = NIFTIDataset(dataset).patient(moving_pat_id)
    regions = arg_to_list(region, str, literals={ 'all': moving_pat.list_regions() })
    region_data = {}
    if regions is not None: 
        for region in regions:
            if not moving_pat.has_region(region):
                if region_ignore_missing:
                    continue
                else:
                    raise ValueError(f"Requested region '{region}' not found for patient '{moving_pat_id}', dataset '{dataset}'.")

            filepath = os.path.join(config.directories.registrations, 'data', dataset, f'{moving_pat_id}-{fixed_pat_id}', f'{region}.npz')
            data = np.load(filepath)['data']
            region_data[region] = data

    return ct_data, region_data
