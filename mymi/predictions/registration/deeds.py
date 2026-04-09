from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, create_affine
import numpy as np
import os
import SimpleITK as sitk
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

from dicomset.utils.geometry import centre_of_mass, foreground_fov
from dicomset.utils.logging import logger
from dicomset.utils.io import load_nifti, save_nifti
from mymi.transforms import crop_or_pad, resample
from mymi.utils.sitk import dvf_to_sitk_transform

def register_deeds(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    fixed_lung_mask: LabelImage3D | None = None,
    moving_lung_mask: LabelImage3D | None = None,
    preprocess_images: bool = True,
    crop_margin_mm: float = 10,
    keep_temp: bool = False,
    ) -> sitk.Transform:
    assert np.all(moving_affine == fixed_affine), "Fixed/moving affines must match."
    affine = fixed_affine
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    if preprocess_images:
        assert fixed_lung_mask is not None and moving_lung_mask is not None, \
            "Lung masks are required when preprocess_images=True."

    temp_dir = tempfile.mkdtemp()
    if keep_temp:
        logger.info(f"Temp directory: {temp_dir}")

    if preprocess_images:
        if len(np.unique(spacing)) > 1:
            model_spacing = (1, 1, 1)
            model_affine = create_affine(model_spacing, origin)
            logger.info(f"Applying isotropic resampling from {spacing} to {model_spacing}.")

            fixed_data = resample(fixed_ct, affine=affine, output_affine=model_affine)
            fixed_label = resample(fixed_lung_mask, affine=affine, output_affine=model_affine)
            moving_data = resample(moving_ct, affine=affine, output_affine=model_affine)
            moving_label = resample(moving_lung_mask, affine=affine, output_affine=model_affine)
        else:
            model_spacing = spacing
            model_affine = create_affine(model_spacing, origin)
            fixed_data = fixed_ct
            fixed_label = fixed_lung_mask
            moving_data = moving_ct
            moving_label = moving_lung_mask

        # Translate moving image to centre lung COMs.
        fixed_com = centre_of_mass(fixed_label, affine=model_affine)
        moving_com = centre_of_mass(moving_label, affine=model_affine)
        trans_mm = np.array(moving_com) - fixed_com
        trans_mm = tuple(trans_mm.astype(np.float64))
        logger.info(f"Translating ({trans_mm}) moving image to align COMs.")
        translate = sitk.TranslationTransform(3)
        translate.SetOffset(trans_mm)
        moving_data, moving_label = resample([moving_data, moving_label], affine=model_affine, transform=translate)

        # Crop to margin surrounding lung.
        logger.info(f"Cropping to {crop_margin_mm}mm surrounding fixed lung mask.")
        fov_min, fov_max = foreground_fov(fixed_label, affine=model_affine)
        fov_min = tuple(np.array(fov_min) - (crop_margin_mm / np.array(model_spacing)))
        fov_max = tuple(np.array(fov_max) + (crop_margin_mm / np.array(model_spacing)))
        crop_mm = (fov_min, fov_max)
        fixed_data, inv_crop = crop_or_pad(fixed_data, crop_mm, affine=model_affine, return_inverse=True)
        fixed_label = crop_or_pad(fixed_label, crop_mm, affine=model_affine)
        moving_data = crop_or_pad(moving_data, crop_mm, affine=model_affine)

        # Save preprocessed files.
        fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
        save_nifti(fixed_data, model_affine, fixed_path)
        fixed_label_path = os.path.join(temp_dir, 'fixed_label.nii.gz')
        save_nifti(fixed_label, model_affine, fixed_label_path)
        moving_path = os.path.join(temp_dir, 'moving.nii.gz')
        save_nifti(moving_data, model_affine, moving_path)
    else:
        model_spacing = spacing

        # Save raw files.
        fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
        save_nifti(fixed_ct, affine, fixed_path)
        moving_path = os.path.join(temp_dir, 'moving.nii.gz')
        save_nifti(moving_ct, affine, moving_path)

    # Run deeds.
    output_prefix = os.path.join(temp_dir, 'output')
    command = [
        'deedsBCV',
        '-F', fixed_path,
        '-M', moving_path,
        '-O', output_prefix,
    ]
    logger.info(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Deeds failed with return code {result.returncode}")

    # Load DVF.
    dvf_path = os.path.join(temp_dir, 'output_dense_disp.nii.gz')
    if not os.path.exists(dvf_path):
        raise RuntimeError(f"Deeds failed — DVF not found at '{dvf_path}'.")

    dvf, _ = load_nifti(dvf_path)
    dvf = dvf * model_spacing      # Deeds uses voxel coords, convert to mm.
    dvf = np.moveaxis(dvf, -1, 0)
    ndvf = dvf.copy()
    dvf[0], dvf[1] = ndvf[1], ndvf[0]  # Deeds swaps x/y axes.

    if preprocess_images:
        logger.info(f"Reversing crop on DVF.")
        crop_affine = create_affine(model_spacing, crop_mm[0])
        dvf = crop_or_pad(dvf, inv_crop, affine=crop_affine, fill=0)
        logger.info(f"Creating composite transform.")
        transform = sitk.CompositeTransform(3)
        transform.AddTransform(translate)
        dvf_transform = dvf_to_sitk_transform(dvf, affine=model_affine)
        transform.AddTransform(dvf_transform)
    else:
        transform = dvf_to_sitk_transform(dvf, affine=affine)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return transform
