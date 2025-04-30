import os
import subprocess
import sys
import tempfile
import torch
from tqdm import tqdm
from typing import *

VXM_PATH="/home/baclark/code/voxelmorph"
os.environ['VXM_BACKEND'] = 'pytorch'
sys.path.append(VXM_PATH)
from voxelmorph.torch.layers import SpatialTransformer

from mymi import config
from mymi.datasets.nifti import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import crop, resample, sitk_transform_points, to_sitk_transform
from mymi.typing import *
from mymi.utils import *

def create_voxelmorph_predictions(
    dataset: str,
    model: str,
    modelname: str,
    model_spacing: ImageSpacing3D,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_study_id: str = 'study_0',
    register_ct: bool = True,
    register_regions: bool = True,
    regions: Optional[Regions] = 'all',
    splits: Optional[Splits] = None) -> None:
    model_path = os.path.join(config.directories.models, 'voxelmorph', model)
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(splits=splits)

    for p in tqdm(pat_ids):
        print(p)
        pat = set.patient(p)
        
        if register_ct:
            with tempfile.TemporaryDirectory() as temp_dir:
                fixed_study = pat.study(fixed_study_id)
                fixed_spacing = fixed_study.ct_spacing
                moving_study = pat.study(moving_study_id)
                moving_spacing = moving_study.ct_spacing

                # Resample images to model spacing.
                fixed_ct = fixed_study.ct_data
                fixed_ct_resampled = resample(fixed_ct, output_spacing=model_spacing, spacing=fixed_spacing)
                fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
                save_as_nifti(fixed_ct_resampled, model_spacing, fixed_study.ct_offset, fixed_path)
                moving_ct = moving_study.ct_data
                moving_ct_resampled = resample(moving_ct, output_spacing=model_spacing, spacing=moving_spacing)
                moving_path = os.path.join(temp_dir, 'moving.nii.gz')
                save_as_nifti(moving_ct_resampled, model_spacing, moving_study.ct_offset, moving_path)

                # Create output paths.
                moved_path = os.path.join(temp_dir, 'moved.nii.gz')
                warp_path = os.path.join(temp_dir, 'warp.nii.gz')

                # Call voxelmorph script.
                command = [
                    'python', os.path.join(VXM_PATH, 'scripts', 'torch', 'register.py'),
                    '--fixed', fixed_path,
                    '--gpu', '0',
                    '--model', model_path,  
                    '--moving', moving_path,
                    '--moved', moved_path,
                    '--warp', warp_path,
                ]
                print(command)
                subprocess.run(command)

                # Load predictions and resample to original spacing.
                moved_ct, _, _ = load_nifti(moved_path)
                moved_ct = resample(moved_ct, output_spacing=fixed_spacing, spacing=model_spacing)
                warp, _, _ = load_nifti(warp_path)
                print(warp.shape)
                assert warp.shape[0] == 3
                warp = resample(warp, output_spacing=fixed_spacing, spacing=model_spacing)

                # Crop predictions to input size to clean up any resampling rounding errors.
                crop_box = ((0, 0, 0), fixed_ct.shape)
                moved_ct = crop(moved_ct, crop_box)
                warp = crop(warp, crop_box)

                # Save predictions.
                pred_base_path = os.path.join(set.path, 'data', 'predictions', 'registration', p, fixed_study_id, p, moving_study_id)
                moved_path = os.path.join(pred_base_path, 'ct', f'{modelname}.nii.gz')
                save_as_nifti(moved_ct, fixed_spacing, fixed_study.ct_offset, moved_path)
                warp_path = os.path.join(pred_base_path, 'dvf', f'{modelname}.nii.gz')
                save_as_nifti(warp, fixed_spacing, fixed_study.ct_offset, warp_path)
                print(moved_path)
                print(warp_path)

        if regions is not None:
            warp_layer = SpatialTransformer()

            pat = set.patient(p)
            pat_regions = regions_to_list(regions, literals={ 'all': pat.list_regions })
            fixed_study = pat.study(fixed_study_id)
            fixed_spacing = fixed_study.ct_spacing
            fixed_offset = fixed_study.ct_offset
            moving_study = pat.study(moving_study_id)

            # Load the warp. Make it so.
            pred_base_path = os.path.join(set.path, 'data', 'predictions', 'registration', p, fixed_study_id, p, moving_study_id)
            warp_path = os.path.join(pred_base_path, 'dvf', f'{modelname}.nii.gz')
            warp, _, _ = load_nifti(warp_path)
            warp = torch.Tensor(warp).unsqueeze(0)

            # Load labels, apply warp and save. 
            for r in pat_regions:
                if not moving_study.has_regions(r):
                    continue

                # Create moved region label.
                label = moving_study.region_data(regions=r)[r]
                label = torch.Tensor(label).unsqueeze(0).unsqueeze(0)
                moved_label = warp_layer(label, warp)
                moved_label = moved_label.squeeze().squeeze().detach().numpy()
                moved_path = os.path.join(pred_base_path, 'regions', r, f'{modelname}.nii.gz')
                os.makedirs(os.path.dirname(moved_path), exist_ok=True)
                save_as_nifti(moved_label, fixed_spacing, fixed_offset, moved_path)
        
        if landmarks is not None:
            pat = set.patient(p)
            pat_landmarks = arg_to_list(landmarks, Landmark, literals={ 'all': pat.list_landmarks })
            fixed_study = pat.study(fixed_study_id)
            if not fixed_study.has_landmarks(pat_landmarks):
                continue

            # Load fixed landmarks.
            fixed_lms = fixed_study.landmark_data(landmarks=pat_landmarks)

            # Load transform.
            fixed_spacing = fixed_study.ct_spacing
            fixed_offset = fixed_study.ct_offset
            pred_base_path = os.path.join(set.path, 'data', 'predictions', 'registration', p, fixed_study_id, p, moving_study_id)
            warp_path = os.path.join(pred_base_path, 'dvf', f'{modelname}.nii.gz')
            warp, _, _ = load_nifti(warp_path)
            warp_sitk = to_sitk_transform(warp, fixed_spacing, fixed_offset)

            # Transform landmarks.
            fixed_lm_data = fixed_lms[list(range(3))]
            moved_lm_data = sitk_transform_points(fixed_lm_data, warp_sitk)
            moved_lms = fixed_lms.copy()
            moved_lms[list(range(3))] = moved_lm_data

            # Save transformed points.
            filepath = os.path.join(pred_base_path, 'landmarks', f'{modelname}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_csv(moved_lms, filepath)
