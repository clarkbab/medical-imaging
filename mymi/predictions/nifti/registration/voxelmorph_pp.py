import os
from tqdm import tqdm
from typing import *

from dicomset.nifti import NiftiDataset
from dicomset.utils import create_affine
from mymi import logging
from mymi.predictions.registration import register_voxelmorph_pp
from mymi.regions import regions_to_list
from mymi.transforms import sitk_load_transform, resample, sitk_save_transform, sitk_transform_points
from mymi.typing import *
from mymi.utils.args import arg_to_list
from mymi.utils.io import save_csv
from mymi.utils.nifti import save_nifti

VXM_PP_PATH = os.path.join(os.environ['CODE'], 'VoxelMorphPlusPlus')

def create_voxelmorph_pp_predictions(
    dataset: str,
    crop_to_lung_centres: bool = True,
    fixed_study: str = 'study_1',
    landmarks: Optional[LandmarkIDs] = 'all',
    lung_region: Region = 'Lungs',
    moving_study: str = 'study_0',
    pat_ids: PatientIDs = 'all',
    perform_breath_resample: bool = False,
    register_ct: bool = True,
    regions: Optional[Regions] = 'all',
    splits: Splits = 'all') -> None:
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids):
        print(p)
        pat = set.patient(p)
        pat_regions = regions_to_list(regions, literals={ 'all': pat.list_regions })
        pat_landmarks = arg_to_list(landmarks, Landmark, literals={ 'all': pat.list_landmarks })
        fixed_study = pat.study(fixed_study)
        moving_study = pat.study(moving_study)
        pred_base_path = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study.id, p, moving_study.id)
        transform_path = os.path.join(pred_base_path, 'transform', f'voxelmorph-pp.hdf5')
        
        if register_ct:
            # Prepare data.
            fixed_ct = fixed_study.ct_data
            moving_ct = moving_study.ct_data
            fixed_affine = create_affine(fixed_study.ct_spacing, fixed_study.ct_origin)
            moving_affine = create_affine(moving_study.ct_spacing, moving_study.ct_origin)
            fixed_lung = fixed_study.regions_data(regions=lung_region)[lung_region]
            moving_lung = moving_study.regions_data(regions=lung_region)[lung_region]

            # Core registration.
            transform = register_voxelmorph_pp(
                fixed_ct, moving_ct, fixed_affine, moving_affine,
                fixed_lung_mask=fixed_lung,
                moving_lung_mask=moving_lung,
                vxm_pp_path=VXM_PP_PATH,
                crop_to_lung_centres=crop_to_lung_centres,
                perform_breath_resample=perform_breath_resample)

            # Save transform.
            os.makedirs(os.path.dirname(transform_path), exist_ok=True)
            sitk_save_transform(transform, transform_path)

            # Save moved CT.
            moved_ct = resample(moving_ct, affine=moving_affine, output_affine=fixed_affine, transform=transform)
            filepath = os.path.join(pred_base_path, 'ct', f'voxelmorph-pp.nii.gz')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_nifti(moved_ct, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

        if regions is not None:
            transform = sitk_load_transform(transform_path)

            for r in pat_regions:
                if not moving_study.has_region(r):
                    continue

                moving_label = moving_study.regions_data(regions=r)[r]
                moved_label = resample(moving_label, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                moved_path = os.path.join(pred_base_path, 'regions', r, f'voxelmorph-pp.nii.gz')
                os.makedirs(os.path.dirname(moved_path), exist_ok=True)
                save_nifti(moved_label, moved_path, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)
        
        if landmarks is not None:
            if not fixed_study.has_landmark(pat_landmarks):
                continue

            transform = sitk_load_transform(transform_path)
            fixed_lms = fixed_study.landmarks_data(landmarks=pat_landmarks)
            fixed_lm_data = fixed_lms[list(range(3))]
            moved_lm_data = sitk_transform_points(fixed_lm_data, transform)
            moved_lms = fixed_lms.copy()
            moved_lms[list(range(3))] = moved_lm_data

            filepath = os.path.join(pred_base_path, 'landmarks', f'voxelmorph-pp.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_csv(moved_lms, filepath)
