import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.datasets.nifti import recreate
from mymi.geometry import get_centre_of_mass
from mymi.transforms import crop_or_pad, crop_or_pad_landmarks, resample
from mymi.utils import *

def create_vxmpp_preprocessed_dataset(
    dataset: str,
    new_dataset: str,
    hu_range: Tuple[float, float] = (-2000, 2000),
    lung_region: Region = 'Lungs',
    pat_ids: PatientIDs = 'all',) -> None:
    set = NiftiDataset(dataset)
    new_set = recreate(new_dataset)
    pat_ids = set.list_patients(ids=pat_ids)

    for p in tqdm(pat_ids):
        # Load data.
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        fixed_ct = fixed_study.ct_data
        fixed_lung = fixed_study.region_data(regions=lung_region)[lung_region]
        fixed_lms = fixed_study.landmark_data()
        moving_study = pat.study('study_0')
        moving_ct = moving_study.ct_data
        moving_lung = moving_study.region_data(regions=lung_region)[lung_region]
        moving_lms = moving_study.landmark_data()

        # Resample to required spacing.
        vxm_fixed_spacing = (1.75, 1.25, 1.75)
        vxm_moving_spacing = (1.75, 1, 1.25)
        fixed_ct_rs = resample(fixed_ct, output_spacing=vxm_fixed_spacing, spacing=fixed_study.ct_spacing)
        fixed_lung_rs = resample(fixed_lung, output_spacing=vxm_fixed_spacing, spacing=fixed_study.ct_spacing)
        moving_ct_rs = resample(moving_ct, output_spacing=vxm_moving_spacing, spacing=moving_study.ct_spacing)
        moving_lung_rs = resample(moving_lung, output_spacing=vxm_moving_spacing, spacing=moving_study.ct_spacing)

        # Crop/pad to required size.
        vxm_size = (192, 192, 208)
        fixed_com = get_centre_of_mass(fixed_lung_rs, use_patient_coords=False)
        moving_com = get_centre_of_mass(moving_lung_rs, use_patient_coords=False)
        half_size = (np.array(vxm_size) / 2).astype(int)
        fixed_crop = (tuple(fixed_com - half_size), tuple(fixed_com + half_size))
        moving_crop = (tuple(moving_com - half_size), tuple(moving_com + half_size))
        fixed_ct_cp = crop_or_pad(fixed_ct_rs, fixed_crop, use_patient_coords=False)
        fixed_lung_cp = crop_or_pad(fixed_lung_rs, fixed_crop, use_patient_coords=False)
        fixed_lms_cp = crop_or_pad_landmarks(fixed_lms, fixed_crop, spacing=vxm_fixed_spacing, offset=(0, 0, 0), use_patient_coords=False)
        moving_ct_cp = crop_or_pad(moving_ct_rs, moving_crop, use_patient_coords=False)
        moving_lung_cp = crop_or_pad(moving_lung_rs, moving_crop, use_patient_coords=False)
        moving_lms_cp = crop_or_pad_landmarks(moving_lms, moving_crop, spacing=vxm_moving_spacing, offset=(0, 0, 0), use_patient_coords=False)

        # Clamp intensity values.
        fixed_ct_cp = np.clip(fixed_ct_cp, a_min=hu_range[0], a_max=hu_range[1])
        moving_ct_cp = np.clip(moving_ct_cp, a_min=hu_range[0], a_max=hu_range[1])

        # Save images and landmarks.
        pat_path = os.path.join(new_set.path, 'data', 'patients', p)
        filepath = os.path.join(pat_path, 'study_1', 'ct', 'series_0.nii.gz')
        save_nifti(fixed_ct_cp, filepath, spacing=vxm_fixed_spacing)
        filepath = os.path.join(pat_path, 'study_1', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(fixed_lung_cp, filepath, spacing=vxm_fixed_spacing)
        filepath = os.path.join(pat_path, 'study_1', 'landmarks', 'series_1.csv')
        save_csv(fixed_lms_cp, filepath, overwrite=True)
        filepath = os.path.join(pat_path, 'study_0', 'ct', 'series_0.nii.gz')
        save_nifti(moving_ct_cp, filepath, spacing=vxm_moving_spacing)
        filepath = os.path.join(pat_path, 'study_0', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(moving_lung_cp, filepath, spacing=vxm_moving_spacing)
        filepath = os.path.join(pat_path, 'study_0', 'landmarks', 'series_1.csv')
        save_csv(moving_lms_cp, filepath, overwrite=True)
