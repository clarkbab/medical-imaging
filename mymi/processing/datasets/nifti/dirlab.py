import numpy as np
import os
import SimpleITK as sitk
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.datasets.nifti import recreate
from mymi.geometry import get_centre_of_mass_mm, get_extent
from mymi.transforms import crop_or_pad_mm, resample, sitk_transform_points
from mymi.utils import *

def create_corrfield_preprocessed_dataset() -> None:
    dataset = 'DIRLAB-LUNG-COPD'
    new_dataset = 'DIRLAB-LUNG-COPD-CP'
    spacing = (1, 1, 1)
    set = NiftiDataset(dataset)
    new_set = recreate(new_dataset)
    pat_ids = set.list_patients()

    for p in tqdm(pat_ids):
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        moving_study = pat.study('study_0')

        # Resample data.
        fixed_ct_rs = resample(fixed_study.ct_data, output_spacing=spacing, spacing=fixed_study.ct_spacing)
        fixed_lung_rs = resample(fixed_study.region_data(regions='Lung')['Lung'], output_spacing=spacing, spacing=fixed_study.ct_spacing)
        moving_ct_rs = resample(moving_study.ct_data, output_spacing=spacing, spacing=moving_study.ct_spacing)
        moving_lung_rs = resample(moving_study.region_data(regions='Lung')['Lung'], output_spacing=spacing, spacing=moving_study.ct_spacing)
        fixed_lms = fixed_study.landmark_data()
        moving_lms = moving_study.landmark_data()

        # Translate moving image to align lung mask COMs.
        fixed_com = get_centre_of_mass_mm(fixed_lung_rs)
        moving_com = get_centre_of_mass_mm(moving_lung_rs)
        trans_mm = np.array(moving_com) - fixed_com
        trans_mm = tuple(trans_mm.astype(np.float64))
        transform = sitk.TranslationTransform(3)
        transform.SetOffset(trans_mm)
        moving_ct_tr, moving_lung_tr = resample([moving_ct_rs, moving_lung_rs], transform=transform)
        moving_lms_tr = sitk_transform_points(moving_lms, transform.GetInverse())

        # Crop to 10mm surrounding fixed lung mask.
        margin = 10
        ext_min, ext_max = get_extent(fixed_lung_rs)
        ext_min = tuple(np.array(ext_min) - margin)
        ext_max = tuple(np.array(ext_max) + margin)
        crop_mm = (ext_min, ext_max)
        fixed_ct_cp = crop_or_pad_mm(fixed_ct_rs, crop_mm)
        fixed_lung_cp = crop_or_pad_mm(fixed_lung_rs, crop_mm)
        moving_ct_cp = crop_or_pad_mm(moving_ct_tr, crop_mm)
        moving_lung_cp = crop_or_pad_mm(moving_lung_tr, crop_mm)

        # Save images.
        pat_path = os.path.join(new_set.path, 'data', 'patients', p)
        filepath = os.path.join(pat_path, 'study_0', 'ct', 'series_0.nii.gz')
        save_nifti(moving_ct_cp, filepath, offset=crop_mm[0], spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'ct', 'series_0.nii.gz')
        save_nifti(fixed_ct_cp, filepath, offset=crop_mm[0], spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(fixed_lung_cp, filepath, offset=crop_mm[0], spacing=spacing)
        filepath = os.path.join(pat_path, 'study_0', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(moving_lung_cp, filepath, offset=crop_mm[0], spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'landmarks', 'series_1.csv')
        save_csv(fixed_lms, filepath, overwrite=True)
        filepath = os.path.join(pat_path, 'study_0', 'landmarks', 'series_1.csv')
        save_csv(moving_lms_tr, filepath, overwrite=True)
