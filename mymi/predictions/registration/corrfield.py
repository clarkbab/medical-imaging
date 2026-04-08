from dicomset.typing import *
from dicomset.utils import affine_origin, affine_spacing, create_affine
import numpy as np
import os
import SimpleITK as sitk
import subprocess
import shutil
import tempfile
from tqdm import tqdm
from typing import Optional, Tuple

from dicomset import NiftiDataset
from dicomset.utils.geometry import centre_of_mass, foreground_fov
from dicomset.utils.logging import logger
from dicomset.utils.io import load_nifti, save_nifti
from mymi.regions import regions_to_list
from mymi.transforms import crop_or_pad, dvf_to_sitk_transform, resample, sitk_save_transform, sitk_transform_points
from mymi.utils.sitk import dvf_to_sitk_transform
from mymi.utils.timer import Timer
from mymi.utils.utils import is_windows

def register_corrfield(
    fixed_ct: Image3D,
    moving_ct: Image3D,
    fixed_affine: AffineMatrix3D,
    moving_affine: AffineMatrix3D,
    fixed_lung_mask: LabelImage3D | None = None,
    moving_lung_mask: LabelImage3D | None = None,
    preprocess_images: bool = True,
    crop_margin_mm: float = 10,
    keep_temp: bool = False,
    ) -> Tuple[Image3D, sitk.Transform]:
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
        fixed_label_path = None

    # Run corrfield.
    cf_path = os.path.join(temp_dir, 'corrfield')
    os.makedirs(cf_path, exist_ok=True)
    cf_out_path = os.path.join(cf_path, 'corrfield')
    command = ['python', r'D:\Brett\code\corrfield\corrfield\corrfield.py'] if is_windows() else ['corrfield']
    command += [
        '-F', fixed_path,
        '-M', moving_path,
    ]
    if fixed_label_path is not None:
        command += ['-m', fixed_label_path]
    command += ['-O', cf_out_path]
    logger.info(command)
    subprocess.run(command)

    # Load DVF.
    cf_dvf_path = os.path.join(cf_path, 'corrfield.nii.gz')
    if not os.path.exists(cf_dvf_path):
        raise RuntimeError(f"Corrfield failed — DVF not found at '{cf_dvf_path}'.")

    dvf, _ = load_nifti(cf_dvf_path)
    dvf = dvf * model_spacing      # Corrfield uses voxel coords, convert to mm.
    dvf = np.moveaxis(dvf, -1, 0)

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

    # Create moved image.
    moved_ct = resample(moving_ct, affine=affine, output_affine=affine, transform=transform)

    return moved_ct, transform

# def create_corrfield_predictions(
#     dataset: str,
#     create_moved_dose: bool = True,
#     fixed_study: StudyID = 'study_1',
#     landmarks: Optional[LandmarkIDs] = 'all',
#     lung_region: Region = 'Lungs',
#     model: str = 'corrfield',
#     moving_study: StudyID = 'study_0',
#     pat_ids: PatientIDs = 'all',
#     preprocess_images: bool = True,
#     regions: Optional[RegionIDs] = 'all',
#     save_as_labels: bool = False,
#     splits: Splits = 'all',
#     use_timing: bool = True,
#     ) -> None:

#     # Create timing table.
#     if use_timing:
#         cols = {
#             'dataset': str,
#             'patient-id': str,
#             'device': str
#         }
#         timer = Timer(cols)

#     # Load patient IDs.
#     set = NiftiDataset(dataset)
#     pat_ids = set.list_patients(patient_ids=pat_ids, split_ids=splits)
#     print(pat_ids)

#     for p in tqdm(pat_ids):
#         # Timing table data.
#         data = {
#             'dataset': dataset,
#             'patient-id': p,
#             'device': 'cuda',
#         }
#         with timer.record(data, enabled=use_timing):
#             pat = set.patient(p)
#             fixed_study = pat.study(fixed_study)
#             moving_study = pat.study(moving_study)

#             # Check for isotropic voxel spacing.
#             fixed_spacing = fixed_study.ct_spacing
#             moving_spacing = moving_study.ct_spacing
#             assert fixed_spacing == moving_spacing
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 if preprocess_images:
#                     if len(np.unique(fixed_spacing)) > 1:
#                         # model_spacing = tuple([np.min(fixed_spacing)] * 3)
#                         model_spacing = (1, 1, 1)
#                         logging.info(f"Applying isotropic resampling from {fixed_spacing} to {model_spacing}.")

#                         # Resample to isotropic spacing.
#                         fixed_data = fixed_study.ct_data
#                         fixed_data = resample(fixed_data, origin=fixed_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=model_spacing, spacing=fixed_study.ct_spacing)
#                         fixed_label = fixed_study.regions_data(regions=lung_region)[lung_region]
#                         fixed_label = resample(fixed_label, origin=fixed_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=model_spacing, spacing=fixed_study.ct_spacing)
#                         moving_data = moving_study.ct_data
#                         moving_data = resample(moving_data, origin=moving_study.ct_origin, output_origin=moving_study.ct_origin, output_spacing=model_spacing, spacing=moving_study.ct_spacing)
#                         moving_label = moving_study.regions_data(regions=lung_region)[lung_region]
#                         moving_label = resample(moving_label, origin=moving_study.ct_origin, output_origin=moving_study.ct_origin, output_spacing=model_spacing, spacing=moving_study.ct_spacing)
#                     else:
#                         model_spacing = fixed_study.ct_spacing

#                     # Translate moving image to centre lung COMs.
#                     fixed_com = centre_of_mass(fixed_label, spacing=model_spacing, origin=fixed_study.ct_origin)
#                     moving_com = centre_of_mass(moving_label, spacing=model_spacing, origin=moving_study.ct_origin)
#                     print(fixed_com, moving_com)
#                     trans_mm = np.array(moving_com) - fixed_com
#                     trans_mm = tuple(trans_mm.astype(np.float64))
#                     logging.info(f"Translating ({trans_mm}) moving image to align COMs.")
#                     translate = sitk.TranslationTransform(3)
#                     translate.SetOffset(trans_mm)
#                     moving_data, moving_label = resample([moving_data, moving_label], transform=translate)
                    
#                     # Crop to 10mm surrounding lung.
#                     logging.info(f"Cropping to 10mm surrounding fixed lung mask.")
#                     margin = 10
#                     fov_min, fov_max = foreground_fov(fixed_label, spacing=model_spacing, origin=fixed_study.ct_origin)
#                     print(fov_min, fov_max)
#                     fov_min = tuple(np.array(fov_min) - (margin / np.array(model_spacing)))
#                     fov_max = tuple(np.array(fov_max) + (margin / np.array(model_spacing)))
#                     crop_mm = (fov_min, fov_max)
#                     print(crop_mm)
#                     fixed_data, inv_crop = crop_or_pad(fixed_data, crop_mm, spacing=model_spacing, origin=fixed_study.ct_origin, return_inverse=True)
#                     print(inv_crop)
#                     fixed_label = crop_or_pad(fixed_label, crop_mm, spacing=model_spacing, origin=fixed_study.ct_origin)
#                     moving_data = crop_or_pad(moving_data, crop_mm, spacing=model_spacing, origin=moving_study.ct_origin)
#                     moving_label = crop_or_pad(moving_label, crop_mm, spacing=model_spacing, origin=moving_study.ct_origin)

#                     # Save files.
#                     fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
#                     save_nifti(fixed_data, fixed_path, spacing=model_spacing, origin=fixed_study.ct_origin)
#                     fixed_label_path = os.path.join(temp_dir, 'fixed_label.nii.gz')
#                     save_nifti(fixed_label, fixed_label_path, spacing=model_spacing, origin=fixed_study.ct_origin)
#                     moving_path = os.path.join(temp_dir, 'moving.nii.gz')
#                     save_nifti(moving_data, moving_path, spacing=model_spacing, origin=moving_study.ct_origin)
#                 else:
#                     fixed_path = fixed_study.ct_filepath
#                     moving_path = moving_study.ct_filepath
#                     fixed_label_path = fixed_study.default_regions.filepath(lung_region)
#                     model_spacing = fixed_study.ct_spacing

#                 # Make total seg prediction.
#                 if save_as_labels:
#                     output_path = os.path.join(set.path, 'data', 'patients', p, s, 'regions', 'series_1')
#                 else:
#                     pred_base = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study, p, moving_study)

#                 # Save keypoint correspondences.
#                 if preprocess_images:
#                     cf_path = os.path.join(temp_dir, 'corrfield')
#                 else:
#                     cf_path = os.path.join(pred_base, 'corrfield')
#                 os.makedirs(cf_path, exist_ok=True)
#                 cf_out_path = os.path.join(cf_path, 'corrfield')
#                 if is_windows():
#                     command = [
#                         'python', r'C:\Users\ClarkBrett\OneDrive - Peter Mac\code\corrfield\corrfield\corrfield',
#                         '-F', fixed_path,
#                         '-M', moving_path,
#                         '-m', fixed_label_path,
#                         '-O', cf_out_path,
#                     ]
#                 else:
#                     command = [
#                         'corrfield',
#                         '-F', fixed_path,
#                         '-M', moving_path,
#                         '-m', fixed_label_path,
#                         '-O', cf_out_path,
#                     ]
#                 logging.info(command)
#                 subprocess.run(command)

#                 cf_dvf_path = os.path.join(cf_path, 'corrfield.nii.gz')
#                 print(cf_dvf_path)
#                 if not os.path.exists(cf_dvf_path):
#                     logging.info(f"Corrfield failed for patient {p}.")
#                     continue

#                 dvf, _, _ = load_nifti(cf_dvf_path)
#                 dvf = dvf * model_spacing      # Corrfield uses voxel coords, convert to mm.
#                 dvf = np.moveaxis(dvf, -1, 0)
#                 if preprocess_images:
#                     logging.info(f"Reversing crop on DVF.")
#                     dvf = crop_or_pad(dvf, inv_crop, spacing=model_spacing, origin=crop_mm[0], fill=0)

#                     logging.info(f"Creating composite transform.")
#                     transform = sitk.CompositeTransform(3)
#                     # Composite mapping from fixed -> warped (dvf) -> translated (COM alignment).
#                     transform.AddTransform(translate)
#                     dvf_transform = dvf_to_sitk_transform(dvf, spacing=model_spacing, origin=fixed_study.ct_origin)
#                     transform.AddTransform(dvf_transform)
#                 else:
#                     transform = dvf_to_sitk_transform(dvf, fixed_study.ct_spacing, fixed_study.ct_origin)
#                 filepath = os.path.join(pred_base, 'transform', f'{model}.hdf5')
#                 sitk_save_transform(transform, filepath)

#                 # Create moved image.
#                 moved_ct = resample(moving_study.ct_data, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
#                 filepath = os.path.join(pred_base, 'ct', f'{model}.nii.gz')
#                 save_nifti(moved_ct, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

#                 if regions is not None:
#                     pat_regions = regions_to_list(regions, literals={ 'all': moving_study.list_regions })
#                     for r in pat_regions:
#                         # Perform transform.
#                         moving_label = moving_study.regions_data(regions=r)[r]
#                         moved_label = resample(moving_label, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
#                         filepath = os.path.join(pred_base, 'regions', r, f'{model}.nii.gz')
#                         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#                         save_nifti(moved_label, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

#                 if landmarks is not None:
#                     fixed_lms_df = fixed_study.landmarks_data(landmarks=landmarks)
#                     if fixed_lms_df is not None:
#                         fixed_lms = fixed_lms_df[list(range(3))].to_numpy()
#                         moved_lms = sitk_transform_points(fixed_lms, transform)
#                         if np.allclose(fixed_lms, moved_lms):
#                             logging.warning(f"Moved points are very similar to fixed points - identity transform?")
#                         moved_lms_df = fixed_lms_df.copy()
#                         moved_lms_df[list(range(3))] = moved_lms
#                         filepath = os.path.join(pred_base, 'landmarks', f'{model}.csv')
#                         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#                         save_csv(moved_lms_df, filepath, overwrite=True)

#                 # Move dose.
#                 if create_moved_dose and moving_study.has_dose:
#                     moving_dose = moving_study.dose_data
#                     moved_dose = resample(moving_dose, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
#                     filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study.id, p, moving_study.id, 'dose', f'{model}.nii.gz')
#                     save_nifti(moved_dose, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

#     # Save timing data.
#     if use_timing:
#         filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
#         timer.save(filepath)
