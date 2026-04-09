from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, create_affine
import itk
import numpy as np
import os
import SimpleITK as sitk
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

from dicomset.utils.logging import logger
from dicomset.utils.io import save_nifti
from mymi.transforms import load_itk_transform
from mymi.utils.itk import from_itk_image
from mymi.utils.sitk import to_sitk_image
from mymi.utils.utils import reverse_xy

def register_unigradicon(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    use_io: bool = False,
    io_iterations: int = 50,
    keep_temp: bool = False,
    ) -> sitk.Transform:
    fixed_spacing = affine_spacing(fixed_affine)
    fixed_origin = affine_origin(fixed_affine)
    moving_spacing = affine_spacing(moving_affine)
    moving_origin = affine_origin(moving_affine)

    temp_dir = tempfile.mkdtemp()
    if keep_temp:
        logger.info(f"Temp directory: {temp_dir}")

    # Save images to temp.
    fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
    save_nifti(fixed_ct, fixed_affine, fixed_path)
    moving_path = os.path.join(temp_dir, 'moving.nii.gz')
    save_nifti(moving_ct, moving_affine, moving_path)
    moved_path = os.path.join(temp_dir, 'moved.nii.gz')
    transform_path = os.path.join(temp_dir, 'transform.hdf5')

    # Run unigradicon.
    io_iter = io_iterations if use_io else None
    command = [
        'unigradicon-register',
        '--moving', moving_path,
        '--moving_modality', 'ct',
        '--fixed', fixed_path,
        '--fixed_modality', 'ct',
        '--warped_moving_out', moved_path,
        '--transform_out', transform_path,
        '--io_iterations', str(io_iter),
    ]
    logger.info(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"UniGradICON failed with return code {result.returncode}")

    # Convert ITK transform to SimpleITK.
    t_itk = load_itk_transform(transform_path)[0]
    transform = _convert_transform_to_sitk(t_itk)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return transform


# Because ITK/SimpleITK load nifti files with negative x/y directions and origins,
# the transform is configured to work with this input space. Which is different from
# how nibabel loads nifti files. ITK expects nifti files to use RAS but we write them
# using LPS.
# Here we flip the transform components to work with nibabel/our coordinate system.
def _convert_transform_to_sitk(t: itk.Transform) -> sitk.Transform:
    # Applied in reverse order in composite transform.
    t0 = itk.down_cast(t.GetNthTransform(0))    # Affine 2 - network to image space
    t1 = itk.down_cast(t.GetNthTransform(1))    # DVF - network to network space
    t2 = itk.down_cast(t.GetNthTransform(2))    # Affine 1 - image to network space

    affine_transforms = [t0, t2]

    new_ts = []
    for at in affine_transforms:
        new_at = _itk_centred_affine_to_sitk(at)
        new_ts.append(new_at)

    # Convert DVF to sitk.
    dvf_image_itk = t1.GetDisplacementField()
    dvf_data, dvf_affine = from_itk_image(dvf_image_itk)
    # Reverse DVF x/y components because our affines map into negative x/y space.
    dvf_data[0], dvf_data[1] = -dvf_data[0], -dvf_data[1]
    dvf_image_sitk = to_sitk_image(dvf_data, affine=dvf_affine, vector=True)
    dir = np.array(dvf_image_sitk.GetDirection())
    dir = reverse_xy(dir)
    dvf_image_sitk.SetDirection(dir)
    dvf_sitk = sitk.DisplacementFieldTransform(dvf_image_sitk)
    new_ts.insert(1, dvf_sitk)

    new_t = sitk.CompositeTransform(new_ts)
    return new_t

def _itk_centred_affine_to_sitk(t: itk.CenteredAffineTransform) -> sitk.AffineTransform:
    dim = t.GetInputSpaceDimension()
    matrix = np.array(t.GetMatrix()).reshape((dim, dim))
    translation = np.array(t.GetTranslation())
    centre = np.array(t.GetCenter())

    # Reverse all x/y params except the scaling.
    # This will map into a network space with negative x/y directions.
    translation[0], translation[1] = -translation[0], -translation[1]
    centre[0], centre[1] = -centre[0], -centre[1]

    sitk_transform = sitk.AffineTransform(dim)
    sitk_transform.SetCenter(centre.tolist())
    sitk_transform.SetMatrix(matrix.flatten().tolist())
    sitk_transform.SetTranslation(translation.tolist())
    return sitk_transform
