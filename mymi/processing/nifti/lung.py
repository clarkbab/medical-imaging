import numpy as np
import os
import shutil
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.datasets.nifti import create, exists, recreate as recreate_nifti
from mymi.geometry import get_centre_of_mass, get_foreground_extent
from mymi.transforms import crop_or_pad, resample
from mymi.utils import *

def create_lung_preprocessed_dataset(
    dataset: str,
    dest_dataset: str,
    hu_range: Tuple[float, float] = (-2000, 2000),
    lung_region: Region = 'Lungs',
    margin_mm: float = 20,
    pat_ids: PatientIDs = 'all',
    recreate: bool = False,
    spacing: Spacing3D = (1, 1, 1)) -> None:
    
    # Load patients.
    set = NiftiDataset(dataset)
    if exists(dest_dataset):
        if recreate:
            dest_set = recreate_nifti(dest_dataset)
        else:
            dest_set = NiftiDataset(dest_dataset)
            # Remove 'patients' data to ensure there are no leftovers. Preserve other data, e.g. predictions, evaluations, etc.
            filepath = os.path.join(dest_set.path, 'data', 'patients')
            if os.path.exists(filepath):
                shutil.rmtree(filepath)
    else:
        dest_set = create(dest_dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids)

    for p in tqdm(pat_ids):
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        moving_study = pat.study('study_0')

        # Resample data.
        fixed_ct_rs = resample(fixed_study.ct_data, output_spacing=spacing, spacing=fixed_study.ct_spacing)
        if fixed_study.has_dose:
            fixed_dose_rs = resample(fixed_study.dose_data, output_spacing=spacing, spacing=fixed_study.dose_spacing)
        fixed_lung_rs = resample(fixed_study.region_images(regions=lung_region)[lung_region], output_spacing=spacing, spacing=fixed_study.ct_spacing)
        moving_ct_rs = resample(moving_study.ct_data, output_spacing=spacing, spacing=moving_study.ct_spacing)
        if moving_study.has_dose:
            moving_dose_rs = resample(moving_study.dose_data, output_spacing=spacing, spacing=moving_study.dose_spacing)
        moving_lung_rs = resample(moving_study.region_images(regions=lung_region)[lung_region], output_spacing=spacing, spacing=moving_study.ct_spacing)
        fixed_lms = fixed_study.landmark_data()
        moving_lms = moving_study.landmark_data()

        # Get COM vector (fixed -> moving).
        fixed_com = get_centre_of_mass(fixed_lung_rs, spacing=spacing, offset=fixed_study.ct_offset)
        moving_com = get_centre_of_mass(moving_lung_rs, spacing=spacing, offset=moving_study.ct_offset)
        trans_mm = np.array(moving_com) - fixed_com

        # Crop to 10mm surrounding fixed lung mask.
        fixed_ext_min, fixed_ext_max = get_foreground_extent(fixed_lung_rs, spacing=spacing, offset=fixed_study.ct_offset)
        fixed_ext_min = tuple(np.array(fixed_ext_min) - margin_mm)
        fixed_ext_max = tuple(np.array(fixed_ext_max) + margin_mm)
        crop_fixed_mm = (fixed_ext_min, fixed_ext_max)
        moving_ext_min = tuple(trans_mm + fixed_ext_min)
        moving_ext_max = tuple(trans_mm + fixed_ext_max)
        crop_moving_mm = (moving_ext_min, moving_ext_max)
        fixed_ct_cp = crop_or_pad(fixed_ct_rs, crop_fixed_mm, spacing=spacing, offset=fixed_study.ct_offset)
        if fixed_study.has_dose:
            fixed_dose_cp = crop_or_pad(fixed_dose_rs, crop_fixed_mm, spacing=spacing, offset=fixed_study.dose_offset)
        fixed_lung_cp = crop_or_pad(fixed_lung_rs, crop_fixed_mm, spacing=spacing, offset=fixed_study.ct_offset)
        moving_ct_cp = crop_or_pad(moving_ct_rs, crop_moving_mm, spacing=spacing, offset=moving_study.ct_offset)
        if moving_study.has_dose:
            moving_dose_cp = crop_or_pad(moving_dose_rs, crop_moving_mm, spacing=spacing, offset=moving_study.dose_offset)
        moving_lung_cp = crop_or_pad(moving_lung_rs, crop_moving_mm, spacing=spacing, offset=moving_study.ct_offset)

        # Move landmarks due to crop (and saving image data with offset=0).
        fixed_lms_data = fixed_lms[list(range(3))]
        fixed_lms_data = fixed_lms_data - fixed_ext_min
        fixed_lms[list(range(3))] = fixed_lms_data
        moving_lms_data = moving_lms[list(range(3))]
        moving_lms_data = moving_lms_data - moving_ext_min
        moving_lms[list(range(3))] = moving_lms_data

        # Clamp intensity values.
        fixed_ct_cp = np.clip(fixed_ct_cp, a_min=hu_range[0], a_max=hu_range[1])
        moving_ct_cp = np.clip(moving_ct_cp, a_min=hu_range[0], a_max=hu_range[1])

        # Save images and landmarks.
        pat_path = os.path.join(dest_set.path, 'data', 'patients', p)
        filepath = os.path.join(pat_path, 'study_1', 'ct', 'series_0.nii.gz')
        save_nifti(fixed_ct_cp, filepath, spacing=spacing)
        if fixed_study.has_dose:
            filepath = os.path.join(pat_path, 'study_1', 'dose', 'series_2.nii.gz')
            save_nifti(fixed_dose_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(fixed_lung_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'landmarks', 'series_1.csv')
        save_csv(fixed_lms, filepath, overwrite=True)
        filepath = os.path.join(pat_path, 'study_0', 'ct', 'series_0.nii.gz')
        save_nifti(moving_ct_cp, filepath, spacing=spacing)
        if moving_study.has_dose:
            filepath = os.path.join(pat_path, 'study_0', 'dose', 'series_2.nii.gz')
            save_nifti(moving_dose_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_0', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(moving_lung_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_0', 'landmarks', 'series_1.csv')
        save_csv(moving_lms, filepath, overwrite=True)
