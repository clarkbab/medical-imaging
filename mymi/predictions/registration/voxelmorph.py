from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, create_affine
import numpy as np
import os
import SimpleITK as sitk
import shutil
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

from dicomset.utils.logging import logger
from dicomset.utils.io import save_nifti
from mymi.transforms import centre_pad, centre_crop, create_sitk_affine_transform, resample
from mymi.utils.nifti import save_nifti as save_nifti_legacy
from mymi.utils.sitk import dvf_to_sitk_transform

def register_voxelmorph(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    model_path: str,
    model_spacing: Spacing3D,
    vxm_path: str,
    pad_shape: Optional[Size3D] = None,
    keep_temp: bool = False,
    ) -> sitk.Transform:
    fixed_spacing = affine_spacing(fixed_affine)
    fixed_origin = affine_origin(fixed_affine)
    moving_spacing = affine_spacing(moving_affine)
    moving_origin = affine_origin(moving_affine)

    model_affine = create_affine(model_spacing, fixed_origin)

    temp_dir = tempfile.mkdtemp()
    if keep_temp:
        logger.info(f"Temp directory: {temp_dir}")

    # Resample images to model spacing.
    fixed_ct_resampled = resample(fixed_ct, affine=fixed_affine, output_affine=model_affine)
    moving_ct_resampled = resample(moving_ct, affine=moving_affine, output_affine=model_affine)

    # Pad images if required.
    if pad_shape is not None:
        resampled_size = fixed_ct_resampled.shape
        fixed_ct_resampled = centre_pad(fixed_ct_resampled, pad_shape, fill=-2000, use_world_coords=False)
        moving_ct_resampled = centre_pad(moving_ct_resampled, pad_shape, fill=-2000, use_world_coords=False)

    # Save files for voxelmorph.
    fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
    save_nifti_legacy(fixed_ct_resampled, fixed_path, spacing=model_spacing, origin=fixed_origin)
    moving_path = os.path.join(temp_dir, 'moving.nii.gz')
    save_nifti_legacy(moving_ct_resampled, moving_path, spacing=model_spacing, origin=moving_origin)
    moved_path = os.path.join(temp_dir, 'moved.npz')
    dvf_path = os.path.join(temp_dir, 'dvf.npz')

    # Run voxelmorph.
    command = [
        sys.executable, os.path.join(vxm_path, 'scripts', 'torch', 'register.py'),
        '--fixed', fixed_path,
        '--gpu', '0',
        '--model', model_path,
        '--moving', moving_path,
        '--moved', moved_path,
        '--warp', dvf_path,
    ]
    logger.info(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"VoxelMorph failed with return code {result.returncode}")

    # Load DVF and create transform.
    model_origin_zero = (0, 0, 0)
    to_model_t = create_sitk_affine_transform(origin=fixed_origin, output_origin=model_origin_zero)
    dvf = np.load(dvf_path)['vol']
    if pad_shape is not None:
        dvf = centre_crop(dvf, resampled_size, use_world_coords=False)

    # VXM DVF is in voxel scale — convert to mm.
    dvf = np.moveaxis(dvf, 0, -1)
    dvf = dvf * np.array(model_spacing)
    dvf = np.moveaxis(dvf, -1, 0)

    dvf_t = dvf_to_sitk_transform(dvf, model_spacing, model_origin_zero)
    to_image_t = create_sitk_affine_transform(origin=model_origin_zero, output_origin=moving_origin)
    transforms = [to_image_t, dvf_t, to_model_t]    # Reverse order.
    transform = sitk.CompositeTransform(transforms)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return transform
