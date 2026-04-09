from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, create_affine
import numpy as np
import os
import SimpleITK as sitk
import shutil
import subprocess
import sys
import tempfile
import torch
from typing import Optional, Tuple

from dicomset.utils.geometry import centre_of_mass
from dicomset.utils.logging import logger
from mymi.transforms import crop_or_pad, resample, sitk_save_transform
from mymi.utils.nifti import save_nifti as save_nifti_legacy
from mymi.utils.sitk import dvf_to_sitk_transform

def register_voxelmorph_pp(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    fixed_lung_mask: LabelImage3D,
    moving_lung_mask: LabelImage3D,
    vxm_pp_path: str,
    crop_to_lung_centres: bool = True,
    perform_breath_resample: bool = False,
    keep_temp: bool = False,
    ) -> sitk.Transform:
    fixed_spacing = affine_spacing(fixed_affine)
    fixed_origin = affine_origin(fixed_affine)
    moving_spacing = affine_spacing(moving_affine)
    moving_origin = affine_origin(moving_affine)

    temp_dir = tempfile.mkdtemp()
    if keep_temp:
        logger.info(f"Temp directory: {temp_dir}")

    fixed_data = fixed_ct
    fixed_lung = fixed_lung_mask
    moving_data = moving_ct
    moving_lung = moving_lung_mask

    # Resample to required spacing.
    vxm_fixed_spacing = (1.75, 1.25, 1.75)
    vxm_moving_spacing = (1.75, 1, 1.25)
    if perform_breath_resample:
        logger.info('Performing breath resampling...')
        fixed_affine_br = create_affine(vxm_fixed_spacing, fixed_origin)
        moving_affine_br = create_affine(vxm_moving_spacing, moving_origin)
        fixed_data = resample(fixed_data, affine=fixed_affine, output_affine=fixed_affine_br)
        fixed_lung = resample(fixed_lung, affine=fixed_affine, output_affine=fixed_affine_br)
        moving_data = resample(moving_data, affine=moving_affine, output_affine=moving_affine_br)
        moving_lung = resample(moving_lung, affine=moving_affine, output_affine=moving_affine_br)
        fixed_spacing = vxm_fixed_spacing
        moving_spacing = vxm_moving_spacing

    # Crop/pad to required size.
    vxm_size = (192, 192, 208)
    fixed_crop = None
    moving_crop = None
    if crop_to_lung_centres:
        logger.info('Cropping to lung centres...')
        fixed_com = centre_of_mass(fixed_lung, use_world_coords=False)
        moving_com = centre_of_mass(moving_lung, use_world_coords=False)
        half_size = (np.array(vxm_size) / 2).astype(int)
        fixed_crop = (tuple(fixed_com - half_size), tuple(fixed_com + half_size))
        moving_crop = (tuple(moving_com - half_size), tuple(moving_com + half_size))
        fixed_data = crop_or_pad(fixed_data, fixed_crop, use_world_coords=False)
        fixed_lung = crop_or_pad(fixed_lung, fixed_crop, use_world_coords=False)
        moving_data = crop_or_pad(moving_data, moving_crop, use_world_coords=False)
        moving_lung = crop_or_pad(moving_lung, moving_crop, use_world_coords=False)

    # Save files.
    fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
    save_nifti_legacy(fixed_data, fixed_path)
    fixed_lung_path = os.path.join(temp_dir, 'fixed-lung.nii.gz')
    save_nifti_legacy(fixed_lung, fixed_lung_path)
    moving_path = os.path.join(temp_dir, 'moving.nii.gz')
    save_nifti_legacy(moving_data, moving_path)
    moving_lung_path = os.path.join(temp_dir, 'moving-lung.nii.gz')
    save_nifti_legacy(moving_lung, moving_lung_path)
    moved_path = os.path.join(temp_dir, 'moved.nii.gz')
    dvf_path = os.path.join(temp_dir, 'dvf.pth')

    # Run VoxelMorph++.
    script_path = os.path.join(vxm_pp_path, 'src', 'inference_voxelmorph_plusplus.py')
    model_path = os.path.join(vxm_pp_path, 'data', 'repeat_l2r_voxelmorph_heatmap_keypoint_fold1.pth')
    command = [
        sys.executable, script_path,
        '--disp_file', dvf_path,
        '--fixed_file', fixed_path,
        '--fixed_mask_file', fixed_lung_path,
        '--moving_file', moving_path,
        '--moving_mask_file', moving_lung_path,
        '--net_model_file', model_path,
        '--warped_file', moved_path,
    ]
    logger.info(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"VoxelMorph++ failed with return code {result.returncode}")

    # Create composite transform that handles fixed/moving resampling and cropping.
    dvf = torch.load(dvf_path)
    dvf = 2 * dvf[0].numpy()    # Resulting DVF is 2x downsampled.
    dvf_transform = dvf_to_sitk_transform(dvf, spacing=(2, 2, 2))
    transform = sitk.CompositeTransform(3)
    moving_br_trans = sitk.AffineTransform(3)
    matrix = np.diag(moving_spacing)
    moving_br_trans.SetMatrix(list(matrix.flatten()))
    transform.AddTransform(moving_br_trans)
    if crop_to_lung_centres and moving_crop is not None:
        moving_clc_trans = sitk.TranslationTransform(3)
        moving_clc_trans.SetOffset(tuple(float(c) for c in moving_crop[0]))
        transform.AddTransform(moving_clc_trans)
    transform.AddTransform(dvf_transform)
    if crop_to_lung_centres and fixed_crop is not None:
        fixed_clc_trans = sitk.TranslationTransform(3)
        fixed_clc_trans.SetOffset(tuple(-float(c) for c in fixed_crop[0]))
        transform.AddTransform(fixed_clc_trans)
    fixed_br_trans = sitk.AffineTransform(3)
    matrix = np.diag(1 / np.array(fixed_spacing))
    fixed_br_trans.SetMatrix(list(matrix.flatten()))
    transform.AddTransform(fixed_br_trans)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return transform
