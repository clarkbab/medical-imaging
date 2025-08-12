from ast import literal_eval
import numpy as np
import os
import shutil
from tqdm import tqdm

from mymi.datasets.nifti import NiftiDataset, create, exists, recreate as recreate_nifti
from mymi.datasets.nifti.utils import *
from mymi.geometry import get_centre_of_mass, foreground_fov
from mymi import logging
from mymi.predictions.nifti import load_registration
from mymi.transforms import crop_or_pad, resample, sitk_transform_points
from mymi.typing import *
from mymi.utils import *

def convert_registrations_from_lung_preprocessed(
    dataset: DatasetID,
    dest_dataset: DatasetID,
    models: ModelIDs,
    convert_ct: bool = True,
    convert_dose: bool = True,
    fixed_study_id = 'study_1',
    landmark_ids: LandmarkIDs = 'all',
    moving_study_id = 'study_0',
    pat_ids: PatientIDs = 'all',
    region_ids: RegionIDs = 'all',
    **kwargs) -> None:
    # Load crops files.
    set = NiftiDataset(dataset)
    dest_set = NiftiDataset(dest_dataset)
    filepath = os.path.join(set.path, 'data', 'crops.csv')
    df = load_csv(filepath, parse_cols=['crop'])

    models = arg_to_list(models, ModelID)
    pat_ids = set.list_patients(pat_ids=pat_ids)
    for m in models:
        logging.info(f'Processing model {m} for dataset {dataset} -> {dest_dataset}.')
        for p in tqdm(pat_ids):
            # Load applied crops.
            fixed_crop = df[(df['patient-id'] == p) & (df['study-id'] == fixed_study_id)].iloc[0]['crop']
            moving_crop = df[(df['patient-id'] == p) & (df['study-id'] == moving_study_id)].iloc[0]['crop']

            # Load transform in preprocessed space.
            transform_pp, _, _, _, _ = load_registration(dataset, p, m, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id)
            if transform_pp is None:
                # Some models (e.g. corrfield) failed for some patients.
                continue

            # Create composite transform from fixed -> moving image in unprocessed space.
            transform = sitk.CompositeTransform(3)
            moving_pp_to_moving = sitk.TranslationTransform(3)
            offset = moving_crop[0]
            moving_pp_to_moving.SetOffset(offset)
            transform.AddTransform(moving_pp_to_moving)
            if isinstance(transform_pp, sitk.CompositeTransform):
                for i in range(transform_pp.GetNumberOfTransforms()):
                    transform.AddTransform(transform_pp.GetNthTransform(i))
            else:
                transform.AddTransform(transform_pp)
            fixed_to_fixed_pp = sitk.TranslationTransform(3)
            offset = to_tuple(-np.array(fixed_crop[0]))
            fixed_to_fixed_pp.SetOffset(offset)
            transform.AddTransform(fixed_to_fixed_pp)

            # Save DVF.
            create_registration_transform(dest_dataset, p, m, transform, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, **kwargs)

            # Save moved CT.
            pat = dest_set.patient(p)
            fixed_study = pat.study(fixed_study_id)
            moving_study = pat.study(moving_study_id) 
            if convert_ct and moving_study.has_ct:
                okwargs = dict(
                    image=moving_study.default_ct,
                    output_image=fixed_study.default_ct,
                    transform=transform,
                )
                moved_ct = resample(**okwargs)
                create_registration_moved_image(dest_dataset, p, m, moved_ct, 'ct', fixed_study.ct_spacing, fixed_study.ct_offset, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, **kwargs)

            # Move region data.
            if region_ids is not None:
                moving_region_data = moving_study.region_data(region_ids=region_ids)
                if moving_region_data is not None:
                    for r, l in moving_region_data.items():
                        # Apply registration transform.
                        okwargs = dict(
                            spacing=moving_study.ct_spacing,
                            offset=moving_study.ct_offset,
                            output_image=fixed_study.default_ct,
                            transform=transform,
                        )
                        moved_label = resample(l, **okwargs)
                        create_registration_moved_region(dest_dataset, p, r, m, moved_label, fixed_study.ct_spacing, fixed_study.ct_offset, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, **kwargs)

            # Move landmarks.
            if landmark_ids is not None:
                fixed_landmark_data = fixed_study.landmark_data(landmark_ids=landmark_ids)
                if fixed_landmark_data is not None:
                    # Move landmarks from fixed -> moving spacing, we can't always invert DVF transforms.
                    moved_landmark_data = sitk_transform_points(fixed_landmark_data, transform)
                    landmark_cols = ['landmark-id', 0, 1, 2]    # Don't save patient-id/study-id cols.
                    moved_landmark_data = moved_landmark_data[landmark_cols]
                    create_registration_moved_landmarks(dest_dataset, p, m, moved_landmark_data, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, **kwargs)

            # Move dose.
            if convert_dose and moving_study.has_dose:
                okwargs = dict(
                    image=moving_study.default_dose,
                    output_image=fixed_study.default_ct,
                    transform=transform,
                )
                moved_dose = resample(**okwargs)
                create_registration_moved_image(dest_dataset, p, m, moved_dose, 'dose', fixed_study.ct_spacing, fixed_study.ct_offset, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, **kwargs)

def convert_to_lung_preprocessed_dataset(
    dataset: str,
    dest_dataset: str,
    hu_range: Tuple[float, float] = (-2000, 2000),
    lung_region: Region = 'Lungs',
    margin_mm: float = 20,
    pat_ids: PatientIDs = 'all',
    recreate: bool = False,
    recreate_patients: bool = True,
    spacing: Spacing3D = (1, 1, 1)) -> None:
    
    # Load patients.
    set = NiftiDataset(dataset)
    if exists(dest_dataset):
        if recreate:
            dest_set = recreate_nifti(dest_dataset)
        else:
            dest_set = NiftiDataset(dest_dataset)
            if recreate_patients:
                # Remove 'patients' data to ensure there are no leftovers. Preserve other data, e.g. predictions, evaluations, etc.
                filepath = os.path.join(dest_set.path, 'data', 'patients')
                if os.path.exists(filepath):
                    shutil.rmtree(filepath)
    else:
        dest_set = create(dest_dataset)

    # Copy required files.
    files = ['index.csv', 'splits.csv']
    for f in files:
        srcpath = os.path.join(set.path, f)
        if os.path.exists(srcpath):
            destpath = os.path.join(dest_set.path, f)
            shutil.copyfile(srcpath, destpath)

    # Load patients.
    pat_ids = set.list_patients(pat_ids=pat_ids)

    # Save the crop that was applied, so that it can be reversed later to apply transforms to original images.
    filepath = os.path.join(dest_set.path, 'data', 'crops.csv')
    if recreate or recreate_patients or not os.path.exists(filepath):
        cols = {
            'patient-id': str,
            'study-id': str,
            'crop': str,
        }
        crops_df = pd.DataFrame(columns=cols.keys())
    else:
        crops_df = load_csv(filepath)

    for p in tqdm(pat_ids):
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        moving_study = pat.study('study_0')

        # Resample data.
        fixed_ct_rs = resample(fixed_study.ct_data, output_spacing=spacing, spacing=fixed_study.ct_spacing)
        if fixed_study.has_dose:
            fixed_dose_rs = resample(fixed_study.dose_data, output_spacing=spacing, spacing=fixed_study.dose_spacing)
        fixed_lung_rs = resample(fixed_study.region_data(regions=lung_region)[lung_region], output_spacing=spacing, spacing=fixed_study.ct_spacing)
        moving_ct_rs = resample(moving_study.ct_data, output_spacing=spacing, spacing=moving_study.ct_spacing)
        if moving_study.has_dose:
            moving_dose_rs = resample(moving_study.dose_data, output_spacing=spacing, spacing=moving_study.dose_spacing)
        moving_lung_rs = resample(moving_study.region_data(regions=lung_region)[lung_region], output_spacing=spacing, spacing=moving_study.ct_spacing)
        fixed_lms = fixed_study.landmark_data()
        moving_lms = moving_study.landmark_data()

        # Get COM vector (fixed -> moving).
        fixed_com = get_centre_of_mass(fixed_lung_rs, spacing=spacing, offset=fixed_study.ct_offset)
        moving_com = get_centre_of_mass(moving_lung_rs, spacing=spacing, offset=moving_study.ct_offset)
        trans_mm = np.array(moving_com) - fixed_com

        # Crop to 10mm surrounding fixed lung mask.
        fixed_fov_min, fixed_fov_max = foreground_fov(fixed_lung_rs, spacing=spacing, offset=fixed_study.ct_offset)
        fixed_fov_min = tuple(float(f) for f in np.array(fixed_fov_min) - margin_mm)
        fixed_fov_max = tuple(float(f) for f in np.array(fixed_fov_max) + margin_mm)
        crop_fixed_mm = (fixed_fov_min, fixed_fov_max)
        moving_fov_min = tuple(float(f) for f in np.array(trans_mm) + fixed_fov_min)
        moving_fov_max = tuple(float(f) for f in np.array(trans_mm) + fixed_fov_max)
        crop_moving_mm = (moving_fov_min, moving_fov_max)
        fixed_ct_cp = crop_or_pad(fixed_ct_rs, crop_fixed_mm, spacing=spacing, offset=fixed_study.ct_offset)
        if fixed_study.has_dose:
            fixed_dose_cp = crop_or_pad(fixed_dose_rs, crop_fixed_mm, spacing=spacing, offset=fixed_study.dose_offset)
        fixed_lung_cp = crop_or_pad(fixed_lung_rs, crop_fixed_mm, spacing=spacing, offset=fixed_study.ct_offset)
        moving_ct_cp = crop_or_pad(moving_ct_rs, crop_moving_mm, spacing=spacing, offset=moving_study.ct_offset)
        if moving_study.has_dose:
            moving_dose_cp = crop_or_pad(moving_dose_rs, crop_moving_mm, spacing=spacing, offset=moving_study.dose_offset)
        moving_lung_cp = crop_or_pad(moving_lung_rs, crop_moving_mm, spacing=spacing, offset=moving_study.ct_offset)

        # Add crop data.
        data = {
            'patient-id': p,
            'study-id': fixed_study.id,
            'crop': str(crop_fixed_mm),
        }
        crops_df = append_row(crops_df, data)
        data = {
            'patient-id': p,
            'study-id': moving_study.id,
            'crop': str(crop_moving_mm),
        }
        crops_df = append_row(crops_df, data)

        # Move landmarks due to crop (and saving image data with offset=0).
        fixed_lms_data = fixed_lms[list(range(3))]
        fixed_lms_data = fixed_lms_data - fixed_fov_min
        fixed_lms[list(range(3))] = fixed_lms_data
        moving_lms_data = moving_lms[list(range(3))]
        moving_lms_data = moving_lms_data - moving_fov_min
        moving_lms[list(range(3))] = moving_lms_data

        # Remove landmarks that are outside the cropped FOV.
        fixed_fov = np.array(fixed_ct_cp.shape) * spacing
        moving_fov = np.array(moving_ct_cp.shape) * spacing
        for a in range(3):
            # Log filtered landmarks.
            fixed_lms_filt = fixed_lms[(fixed_lms[a] < 0) | (fixed_lms[a] >= fixed_fov[a])]
            if len(fixed_lms_filt) > 0:
                landmarks_str = ','.join(fixed_lms_filt['landmark-id'].values)
                logging.warning(f"Filtering landmarks '{landmarks_str}' from patient {p} study {fixed_study.id} due to cropping.")
            moving_lms_filt = moving_lms[(moving_lms[a] < 0) | (moving_lms[a] >= moving_fov[a])]
            if len(moving_lms_filt) > 0:
                landmarks_str = ','.join(moving_lms_filt['landmark-id'].values)
                logging.warning(f"Filtering landmarks '{landmarks_str}' from patient {p} study {moving_study.id} due to cropping.")

            # Filter landmarks.
            fixed_lms = fixed_lms[(fixed_lms[a] >= 0) & (fixed_lms[a] < fixed_fov[a])]
            moving_lms = moving_lms[(moving_lms[a] >= 0) & (moving_lms[a] < fixed_fov[a])]

        # Clamp intensity values.
        fixed_ct_cp = np.clip(fixed_ct_cp, a_min=hu_range[0], a_max=hu_range[1])
        moving_ct_cp = np.clip(moving_ct_cp, a_min=hu_range[0], a_max=hu_range[1])

        # Save images and landmarks.
        pat_path = os.path.join(dest_set.path, 'data', 'patients', p)
        filepath = os.path.join(pat_path, 'study_1', 'ct', 'series_0.nii.gz')
        save_nifti(fixed_ct_cp, filepath, spacing=spacing)
        if fixed_study.has_dose:
            filepath = os.path.join(pat_path, 'study_1', 'dose', 'series_2.nii.gz')
            save_nifti(fixed_dose_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(fixed_lung_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_1', 'landmarks', 'series_1.csv')
        save_csv(fixed_lms, filepath, overwrite=True)
        filepath = os.path.join(pat_path, 'study_0', 'ct', 'series_0.nii.gz')
        save_nifti(moving_ct_cp, filepath, spacing=spacing)
        if moving_study.has_dose:
            filepath = os.path.join(pat_path, 'study_0', 'dose', 'series_2.nii.gz')
            save_nifti(moving_dose_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_0', 'regions', 'series_1', 'Lungs.nii.gz')
        save_nifti(moving_lung_cp, filepath, spacing=spacing)
        filepath = os.path.join(pat_path, 'study_0', 'landmarks', 'series_1.csv')
        save_csv(moving_lms, filepath, overwrite=True)

    # Save crop data.
    filepath = os.path.join(dest_set.path, 'data', 'crops.csv')
    save_csv(crops_df, filepath, overwrite=True)
