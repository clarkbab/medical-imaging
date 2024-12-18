import os
from tqdm import tqdm
from typing import Optional

from mymi.dataset.nifti import NiftiDataset, recreate as recreate_nifti
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import sitk_save_transform
from mymi.transforms.dataset.nifti import rigid_registration
from mymi.types import PatientLandmarks, PatientRegions
from mymi.utils import save_as_nifti, save_csv

from ...processing import write_flag

def create_registered_dataset(
    dataset: str,
    dest_dataset: str,
    landmarks: Optional[PatientLandmarks] = None,
    regions: Optional[PatientRegions] = None) -> None:
    logging.info(f'Registering {dataset} to {dest_dataset}.')

    # Create dest dataset.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions)
    dest_set = recreate_nifti(dest_dataset)
    write_flag(dest_set, f'__REGISTERED_FROM_{dataset}')

    pat_ids = set.list_patients()
    for p in tqdm(pat_ids):
        logging.info(p)
        # Perform rigid registration.
        moved_ct, moved_region_data, moved_landmarks, _ = rigid_registration(dataset, p, 'study_0', p, 'study_1', landmarks=landmarks, regions=regions, regions_ignore_missing=True)
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        fixed_spacing = fixed_study.ct_spacing
        fixed_offset = fixed_study.ct_offset
        moved_study = dest_set.patient(p, check_path=False).study('study_0', check_path=False)
        filepath = os.path.join(moved_study.path, 'ct', 'series_0.nii.gz')
        save_as_nifti(moved_ct, fixed_spacing, fixed_offset, filepath)

        if moved_region_data is not None:
            for r, moved_r in moved_region_data.items():
                filepath = os.path.join(moved_study.path, 'regions', 'series_1', f'{r}.nii.gz')
                save_as_nifti(moved_r, fixed_spacing, fixed_offset, filepath)

        if moved_landmarks is not None:
            filepath = os.path.join(moved_study.path, 'landmarks', 'series_1.csv')
            save_csv(moved_landmarks, filepath)

        # Save transform - we'll need this to propagate fixed landmarks to non-registered moving images.
        # Points can't be propagated from moving -> fixed without the inverse transform. TRE is typically
        # calculated in the moving image space. 
        # Actually, we can easily invert a rigid transform, just propagate points from moving to fixed.
        # filepath = os.path.join(moved_study.path, 'transform.tfm')
        # sitk_save_transform(transform, filepath)

        # Add fixed data.
        fixed_ct = fixed_study.ct_data 
        dest_fixed_study = dest_set.patient(p, check_path=False).study('study_1', check_path=False)
        filepath = os.path.join(dest_fixed_study.path, 'ct', 'series_1.nii.gz')
        save_as_nifti(fixed_ct, fixed_spacing, fixed_offset, filepath)

        if regions is not None:
            fixed_region_data = fixed_study.region_data(regions=regions, regions_ignore_missing=True)
            for r, fixed_r in fixed_region_data.items():
                filepath = os.path.join(dest_fixed_study.path, 'regions', 'series_1', f'{r}.nii.gz')
                save_as_nifti(fixed_r, fixed_spacing, fixed_offset, filepath)

        if landmarks is not None:
            fixed_landmarks = fixed_study.landmark_data(landmarks=landmarks)
            filepath = os.path.join(dest_fixed_study.path, 'landmarks', 'series_1.csv')
            save_csv(fixed_landmarks, filepath)
