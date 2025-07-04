import os
import subprocess
import tempfile
from tqdm import tqdm
from typing import *

from mymi.datasets.nifti import NiftiDataset
from mymi.geometry import get_centre_of_mass
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import *
from mymi.typing import *
from mymi.utils import *

VXM_PP_PATH = os.path.join(os.environ['CODE'], 'VoxelMorphPlusPlus')

def create_voxelmorph_pp_predictions(
    dataset: str,
    crop_to_lung_centres: bool = True,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    lung_region: Region = 'Lungs',
    moving_study_id: str = 'study_0',
    pat_ids: PatientIDs = 'all',
    perform_breath_resample: bool = False,   # Accounts for inhale/exhale differences.
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
        fixed_study = pat.study(fixed_study_id)
        moving_study = pat.study(moving_study_id)
        pred_base_path = os.path.join(set.path, 'data', 'predictions', 'registration', p, fixed_study_id, p, moving_study_id)
        transform_path = os.path.join(pred_base_path, 'dvf', f'voxelmorph-pp.hdf5')
        
        if register_ct:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Load data.
                fixed_ct = fixed_study.ct_data
                fixed_spacing = fixed_study.ct_spacing
                fixed_lung = fixed_study.regions_data(regions=lung_region)[lung_region]
                moving_ct = moving_study.ct_data
                moving_spacing = moving_study.ct_spacing
                moving_lung = moving_study.regions_data(regions=lung_region)[lung_region]

                # Resample to required spacing.
                vxm_fixed_spacing = (1.75, 1.25, 1.75)
                vxm_moving_spacing = (1.75, 1, 1.25)
                if perform_breath_resample:
                    logging.info('Performing breath resampling...')
                    fixed_ct = resample(fixed_ct, output_spacing=vxm_fixed_spacing, spacing=fixed_spacing)
                    fixed_lung = resample(fixed_lung, output_spacing=vxm_fixed_spacing, spacing=fixed_spacing)
                    moving_ct = resample(moving_ct, output_spacing=vxm_moving_spacing, spacing=moving_spacing)
                    moving_lung = resample(moving_lung, output_spacing=vxm_moving_spacing, spacing=moving_spacing)
                    fixed_spacing = vxm_fixed_spacing
                    moving_spacing = vxm_moving_spacing

                # Crop/pad to required size.
                vxm_size = (192, 192, 208)
                if crop_to_lung_centres:
                    logging.info('Cropping to lung centres...')
                    fixed_com = get_centre_of_mass(fixed_lung, use_patient_coords=False)
                    moving_com = get_centre_of_mass(moving_lung, use_patient_coords=False)
                    half_size = (np.array(vxm_size) / 2).astype(int)
                    fixed_crop = (tuple(fixed_com - half_size), tuple(fixed_com + half_size))
                    moving_crop = (tuple(moving_com - half_size), tuple(moving_com + half_size))
                    fixed_ct = crop_or_pad(fixed_ct, fixed_crop, use_patient_coords=False)
                    fixed_lung = crop_or_pad(fixed_lung, fixed_crop, use_patient_coords=False)
                    moving_ct = crop_or_pad(moving_ct, moving_crop, use_patient_coords=False)
                    moving_lung = crop_or_pad(moving_lung, moving_crop, use_patient_coords=False)

                # Save files for 'voxelmorph'.
                fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
                save_nifti(fixed_ct, fixed_path)
                fixed_lung_path = os.path.join(temp_dir, 'fixed-lung.nii.gz')
                save_nifti(fixed_lung, fixed_lung_path)
                moving_path = os.path.join(temp_dir, 'moving.nii.gz')
                save_nifti(moving_ct, moving_path)
                moving_lung_path = os.path.join(temp_dir, 'moving-lung.nii.gz')
                save_nifti(moving_lung, moving_lung_path)
                moved_path = os.path.join(temp_dir, 'moved.nii.gz')
                dvf_path = os.path.join(temp_dir, 'dvf.pth')

                # Call voxelmorph script - for transform only.
                print('Running VoxelMorph++...')
                script_path = os.path.join(VXM_PP_PATH, 'src', 'inference_voxelmorph_plusplus.py')
                model_path = os.path.join(VXM_PP_PATH, 'data', 'repeat_l2r_voxelmorph_heatmap_keypoint_fold1.pth')
                command = [
                    'python', script_path,
                    '--disp_file', dvf_path,
                    '--fixed_file', fixed_path,
                    '--fixed_mask_file', fixed_lung_path,
                    '--moving_file', moving_path,
                    '--moving_mask_file', moving_lung_path,
                    '--net_model_file', model_path,  
                    '--warped_file', moved_path,
                ]
                logging.info(command)
                subprocess.run(command)
                print('VoxelMorph++ finished.')
                print(moved_path)

                # Create composite transform that handles fixed/moving resampling and cropping.
                dvf = torch.load(dvf_path)
                dvf = 2 * dvf[0].numpy()    # Resulting DVF is 2x downsampled.
                dvf_transform = dvf_to_sitk_transform(dvf, spacing=(2, 2, 2))
                transform = sitk.CompositeTransform(3)
                moving_br_trans = sitk.AffineTransform(3)
                matrix = np.diag(moving_spacing)
                moving_br_trans.SetMatrix(list(matrix.flatten()))
                transform.AddTransform(moving_br_trans)
                if crop_to_lung_centres:
                    moving_clc_trans = sitk.TranslationTransform(3)
                    moving_clc_trans.SetOffset((float(c) for c in moving_crop[0]))
                    transform.AddTransform(moving_clc_trans)
                transform.AddTransform(dvf_transform)
                if crop_to_lung_centres:
                    fixed_clc_trans = sitk.TranslationTransform(3)
                    fixed_clc_trans.SetOffset((-float(c) for c in fixed_crop[0]))
                    transform.AddTransform(fixed_clc_trans)
                fixed_br_trans = sitk.AffineTransform(3)
                matrix = np.diag(1 / np.array(fixed_spacing))
                fixed_br_trans.SetMatrix(list(matrix.flatten()))
                transform.AddTransform(fixed_br_trans)
                save_sitk_transform(transform, transform_path)

                # Move image manually using transform - only requires one resampling.
                moved_ct = resample(moving_study.ct_data, offset=moving_study.ct_offset, output_offset=fixed_study.ct_offset, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(pred_base_path, 'ct', f'voxelmorph-pp.nii.gz')
                save_nifti(moved_ct, filepath, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)

        if regions is not None:
            transform = load_sitk_transform(transform_path)

            # Load labels, apply transform and save. 
            for r in pat_regions:
                if not moving_study.has_regions(r):
                    continue

                # Create moved region label.
                moving_label = moving_study.regions_data(regions=r)[r]
                moved_label = resample(moving_label, offset=moving_study.ct_offset, output_offset=fixed_study.ct_offset, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                moved_path = os.path.join(pred_base_path, 'regions', r, f'voxelmorph-pp.nii.gz')
                os.makedirs(os.path.dirname(moved_path), exist_ok=True)
                save_nifti(moved_label, moved_path, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)
        
        if landmarks is not None:
            if not fixed_study.has_landmarks(pat_landmarks):
                continue

            # Load transform and fixed landmarks.
            transform = load_sitk_transform(transform_path)
            fixed_lms = fixed_study.landmarks_data(landmarks=pat_landmarks)

            # Transform landmarks.
            fixed_lm_data = fixed_lms[list(range(3))]
            moved_lm_data = sitk_transform_points(fixed_lm_data, transform)
            moved_lms = fixed_lms.copy()
            moved_lms[list(range(3))] = moved_lm_data

            # Save transformed points.
            filepath = os.path.join(pred_base_path, 'landmarks', f'voxelmorph-pp.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_csv(moved_lms, filepath)
