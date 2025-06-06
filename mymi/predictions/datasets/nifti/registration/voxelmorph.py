import os
import subprocess
import sys
import tempfile
from tqdm import tqdm
from typing import *

VXM_PATH = os.path.join(os.environ['CODE'], 'voxelmorph')
os.environ['VXM_BACKEND'] = 'pytorch'
sys.path.append(VXM_PATH)

from mymi import config
from mymi.datasets.nifti import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import *
from mymi.typing import *
from mymi.utils import *

def create_voxelmorph_predictions(
    dataset: str,
    model: str,
    modelname: str,
    model_spacing: Spacing3D,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_study_id: str = 'study_0',
    pad_shape: Optional[Size3D] = None,
    pat_ids: PatientIDs = 'all',
    register_ct: bool = True,
    regions: Optional[Regions] = 'all',
    splits: Optional[Splits] = None) -> None:
    model_path = os.path.join(config.directories.models, 'voxelmorph', model)
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids):
        print(p)
        pat = set.patient(p)
        pat_regions = regions_to_list(regions, literals={ 'all': pat.list_regions })
        pat_landmarks = arg_to_list(landmarks, Landmark, literals={ 'all': pat.list_landmarks })
        fixed_study = pat.study(fixed_study_id)
        moving_study = pat.study(moving_study_id)
        pred_base_path = os.path.join(set.path, 'data', 'predictions', 'registration', p, fixed_study_id, p, moving_study_id)
        transform_path = os.path.join(pred_base_path, 'dvf', f'{modelname}.hdf5')
        
        if register_ct:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Resample images to model spacing.
                fixed_ct = fixed_study.ct_data
                fixed_ct_resampled = resample(fixed_ct, output_spacing=model_spacing, spacing=fixed_study.ct_spacing)
                moving_ct = moving_study.ct_data
                moving_ct_resampled = resample(moving_ct, output_spacing=model_spacing, spacing=moving_study.ct_spacing)

                # Pad images if required.
                if pad_shape is not None:
                    resampled_size = fixed_ct_resampled.shape
                    fixed_ct_resampled = centre_pad(fixed_ct_resampled, pad_shape, fill=-2000, use_patient_coords=False)
                    moving_ct_resampled = centre_pad(moving_ct_resampled, pad_shape, fill=-2000, use_patient_coords=False)

                # Save files for 'voxelmorph'.
                fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
                save_nifti(fixed_ct_resampled, fixed_path, spacing=model_spacing, offset=fixed_study.ct_offset)
                moving_path = os.path.join(temp_dir, 'moving.nii.gz')
                save_nifti(moving_ct_resampled, moving_path, spacing=model_spacing, offset=moving_study.ct_offset)
                moved_path = os.path.join(temp_dir, 'moved.npz')
                dvf_path = os.path.join(temp_dir, 'dvf.npz')

                # Call voxelmorph script - for transform only.
                print('Running VoxelMorph...')
                command = [
                    'python', os.path.join(VXM_PATH, 'scripts', 'torch', 'register.py'),
                    '--fixed', fixed_path,
                    '--gpu', '0',
                    '--model', model_path,  
                    '--moving', moving_path,
                    '--moved', moved_path,
                    '--warp', dvf_path,
                ]
                logging.info(command)
                subprocess.run(command)
                print('VoxelMorph finished.')
                print(moved_path)

                # Load, resample, save DVF.
                # Transform goes from fixed -> moving image.
                model_offset = (0, 0, 0)
                to_model_t = create_sitk_affine_transform(offset=fixed_study.ct_offset, output_offset=model_offset)
                dvf = np.load(dvf_path)['vol']
                if pad_shape is not None:
                    dvf = centre_crop(dvf, resampled_size, use_patient_coords=False)
                save_numpy(dvf, 'dvf.npz')
                assert dvf.shape[0] == 3
                # VXM DVF is on the scale of image voxels in the image - need to convert to mm.
                dvf = np.moveaxis(dvf, 0, -1)
                dvf = dvf * np.array(model_spacing)  # Convert to mm.
                dvf = np.moveaxis(dvf, -1, 0)
                dvf_t = dvf_to_sitk_transform(dvf, model_spacing, model_offset)
                to_image_t = create_sitk_affine_transform(offset=model_offset, output_offset=moving_study.ct_offset)
                transforms = [to_image_t, dvf_t, to_model_t]    # Reverse order.
                transform = sitk.CompositeTransform(transforms)
                save_sitk_transform(transform, transform_path)

                # Move image manually using transform - only requires one resampling.
                moved_ct = sitk_transform_image(moving_ct, transform, fixed_ct.shape, fill=0, offset=moving_study.ct_offset, output_spacing=fixed_study.ct_spacing, output_offset=fixed_study.ct_offset, spacing=moving_study.ct_spacing)
                # moved_ct = load_numpy(moved_path, keys='vol')
                moved_path = os.path.join(pred_base_path, 'ct', f'{modelname}.nii.gz')
                save_nifti(moved_ct, moved_path, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)

        if regions is not None:
            transform = load_sitk_transform(transform_path)

            # Load labels, apply transform and save. 
            for r in pat_regions:
                if not moving_study.has_regions(r):
                    continue

                # Create moved region label.
                moving_label = moving_study.region_data(regions=r)[r]
                moved_label = sitk_transform_image(moving_label, transform, fixed_study.ct_size, offset=moving_study.ct_offset, output_spacing=fixed_study.ct_spacing, output_offset=fixed_study.ct_offset, spacing=moving_study.ct_spacing)
                moved_path = os.path.join(pred_base_path, 'regions', r, f'{modelname}.nii.gz')
                os.makedirs(os.path.dirname(moved_path), exist_ok=True)
                save_nifti(moved_label, moved_path, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)
        
        if landmarks is not None:
            if not fixed_study.has_landmarks(pat_landmarks):
                continue

            # Load transform and fixed landmarks.
            transform = load_sitk_transform(transform_path)
            fixed_lms = fixed_study.landmark_data(landmarks=pat_landmarks)

            # Transform landmarks.
            fixed_lm_data = fixed_lms[list(range(3))]
            moved_lm_data = sitk_transform_points(fixed_lm_data, transform)
            moved_lms = fixed_lms.copy()
            moved_lms[list(range(3))] = moved_lm_data

            # Save transformed points.
            filepath = os.path.join(pred_base_path, 'landmarks', f'{modelname}.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_csv(moved_lms, filepath)
