import os
import subprocess
import tempfile
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.geometry import get_centre_of_mass, foreground_fov
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import crop_or_pad, dvf_to_sitk_transform, resample, sitk_save_transform, sitk_transform_points
from mymi.typing import *
from mymi.utils import *

def create_deeds_predictions(
    dataset: str,
    create_moved_dose: bool = True,
    fixed_study_id: StudyID = 'study_1',
    landmark_ids: Optional[LandmarkIDs] = 'all',
    lung_region: str = 'Lungs',
    model: str = 'deeds',
    moving_study_id: StudyID = 'study_0',
    pat_ids: PatientIDs = 'all',
    preprocess_images: bool = True,
    region_ids: Optional[RegionIDs] = 'all',
    save_as_labels: bool = False,
    splits: Splits = 'all',
    use_timing: bool = True) -> None:

    # Create timing table.
    if use_timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patient IDs.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids):
        # Timing table data.
        data = {
            'dataset': dataset,
            'patient-id': p,
            'device': 'cuda',
        }
        with timer.record(data, enabled=use_timing):
            pat = set.patient(p)
            fixed_study = pat.study(fixed_study_id)
            moving_study = pat.study(moving_study_id)

            # Check for isotropic voxel spacing.
            fixed_spacing = fixed_study.ct_spacing
            moving_spacing = moving_study.ct_spacing
            assert fixed_spacing == moving_spacing
            with tempfile.TemporaryDirectory() as temp_dir:
                if preprocess_images:
                    if len(np.unique(fixed_spacing)) > 1:
                        # model_spacing = tuple([np.min(fixed_spacing)] * 3)
                        model_spacing = (1, 1, 1)
                        logging.info(f"Applying isotropic resampling from {fixed_spacing} to {model_spacing}.")

                        # Resample to isotropic spacing.
                        fixed_data = fixed_study.ct_data
                        fixed_data = resample(fixed_data, offset=fixed_study.ct_offset, output_offset=fixed_study.ct_offset, output_spacing=model_spacing, spacing=fixed_study.ct_spacing)
                        fixed_label = fixed_study.region_data(region_ids=lung_region)[lung_region]
                        fixed_label = resample(fixed_label, offset=fixed_study.ct_offset, output_offset=fixed_study.ct_offset, output_spacing=model_spacing, spacing=fixed_study.ct_spacing)
                        moving_data = moving_study.ct_data
                        moving_data = resample(moving_data, offset=moving_study.ct_offset, output_offset=moving_study.ct_offset, output_spacing=model_spacing, spacing=moving_study.ct_spacing)
                        moving_label = moving_study.region_data(region_ids=lung_region)[lung_region]
                        moving_label = resample(moving_label, offset=moving_study.ct_offset, output_offset=moving_study.ct_offset, output_spacing=model_spacing, spacing=moving_study.ct_spacing)
                    else:
                        model_spacing = fixed_study.ct_spacing

                    # Translate moving image to centre lung COMs.
                    fixed_com = get_centre_of_mass(fixed_label, spacing=model_spacing, offset=fixed_study.ct_offset)
                    moving_com = get_centre_of_mass(moving_label, spacing=model_spacing, offset=moving_study.ct_offset)
                    print(fixed_com, moving_com)
                    trans_mm = np.array(moving_com) - fixed_com
                    trans_mm = tuple(trans_mm.astype(np.float64))
                    logging.info(f"Translating ({trans_mm}) moving image to align COMs.")
                    translate = sitk.TranslationTransform(3)
                    translate.SetOffset(trans_mm)
                    moving_data, moving_label = resample([moving_data, moving_label], transform=translate)
                    
                    # Crop to 10mm surrounding lung.
                    logging.info(f"Cropping to 10mm surrounding fixed lung mask.")
                    margin = 10
                    fov_min, fov_max = foreground_fov(fixed_label, spacing=model_spacing, offset=fixed_study.ct_offset)
                    print(fov_min, fov_max)
                    fov_min = tuple(np.array(fov_min) - (margin / np.array(model_spacing)))
                    fov_max = tuple(np.array(fov_max) + (margin / np.array(model_spacing)))
                    crop_mm = (fov_min, fov_max)
                    print(crop_mm)
                    fixed_data, inv_crop = crop_or_pad(fixed_data, crop_mm, spacing=model_spacing, offset=fixed_study.ct_offset, return_inverse=True)
                    print(inv_crop)
                    fixed_label = crop_or_pad(fixed_label, crop_mm, spacing=model_spacing, offset=fixed_study.ct_offset)
                    moving_data = crop_or_pad(moving_data, crop_mm, spacing=model_spacing, offset=moving_study.ct_offset)
                    moving_label = crop_or_pad(moving_label, crop_mm, spacing=model_spacing, offset=moving_study.ct_offset)

                    # Save files.
                    fixed_path = os.path.join(temp_dir, 'fixed.nii.gz')
                    save_nifti(fixed_data, fixed_path, spacing=model_spacing, offset=fixed_study.ct_offset)
                    fixed_label_path = os.path.join(temp_dir, 'fixed_label.nii.gz')
                    save_nifti(fixed_label, fixed_label_path, spacing=model_spacing, offset=fixed_study.ct_offset)
                    moving_path = os.path.join(temp_dir, 'moving.nii.gz')
                    save_nifti(moving_data, moving_path, spacing=model_spacing, offset=moving_study.ct_offset)
                else:
                    fixed_path = fixed_study.ct_filepath
                    moving_path = moving_study.ct_filepath
                    fixed_label_path = fixed_study.default_regions.filepath(lung_region)
                    model_spacing = fixed_study.ct_spacing

                # Initial affine alignment.
                # Assume data is already affinely aligned.
                # command = [
                #     'linearBCV',
                #     '-F', fixed_study.ct_path,
                #     '-M', moving_study.ct_path,
                #     '-O', os.path.join(temp_dir, 'affine_2_4'),
                # ]
                # logging.info(command)
                # subprocess.run(command)
                
                # Deformable registration.
                command = [
                    'deedsBCV',
                    '-F', fixed_path,
                    '-M', moving_path,
                    '-O', os.path.join(temp_dir, 'output'),
                ]
                logging.info(command)
                subprocess.run(command)

                if save_as_labels:
                    output_path = os.path.join(set.path, 'data', 'patients', p, s, 'regions', 'series_1')
                else:
                    pred_base = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study_id, p, moving_study_id)

                filepath = os.path.join(temp_dir, 'output_dense_disp.nii.gz')
                dvf, _, _ = load_nifti(filepath)
                dvf = dvf * model_spacing      # Deeds uses voxel coords, convert to mm.
                dvf = np.moveaxis(dvf, -1, 0)
                ndvf = dvf.copy()
                dvf[0], dvf[1] = ndvf[1], ndvf[0]  # Deeds swaps x/y axes.
                if preprocess_images:
                    logging.info(f"Reversing crop on DVF.")
                    dvf = crop_or_pad(dvf, inv_crop, spacing=model_spacing, offset=crop_mm[0], fill=0)

                    logging.info(f"Creating composite transform.")
                    transform = sitk.CompositeTransform(3)
                    # Composite mapping from fixed -> warped (dvf) -> translated (COM alignment).
                    transform.AddTransform(translate)
                    dvf_transform = dvf_to_sitk_transform(dvf, spacing=model_spacing, offset=fixed_study.ct_offset)
                    transform.AddTransform(dvf_transform)
                else:
                    transform = dvf_to_sitk_transform(dvf, fixed_study.ct_spacing, fixed_study.ct_offset)
                filepath = os.path.join(pred_base, 'dvf', f'{model}.hdf5')
                sitk_save_transform(transform, filepath)

                # Create moved image.
                moved_ct = resample(moving_study.ct_data, offset=moving_study.ct_offset, output_offset=fixed_study.ct_offset, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(pred_base, 'ct', f'{model}.nii.gz')
                save_nifti(moved_ct, filepath, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)

                if region_ids is not None:
                    pat_regions = regions_to_list(region_ids, literals={ 'all': pat.list_regions })
                    for r in pat_regions:
                        # Perform transform.
                        moving_label = moving_study.region_data(region_ids=r)[r]
                        moved_label = resample(moving_label, offset=moving_study.ct_offset, output_offset=fixed_study.ct_offset, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                        filepath = os.path.join(pred_base, 'regions', r, f'{model}.nii.gz')
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        save_nifti(moved_label, filepath, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)

                if landmark_ids is not None:
                    fixed_lms_df = fixed_study.landmark_data(landmark_ids=landmark_ids)
                    if fixed_lms_df is not None:
                        fixed_lms = fixed_lms_df[list(range(3))].to_numpy()
                        moved_lms = sitk_transform_points(fixed_lms, transform)
                        if np.allclose(fixed_lms, moved_lms):
                            logging.warning(f"Moved points are very similar to fixed points - identity transform?")
                        moved_lms_df = fixed_lms_df.copy()
                        moved_lms_df[list(range(3))] = moved_lms
                        filepath = os.path.join(pred_base, 'landmarks', f'{model}.csv')
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        save_csv(moved_lms_df, filepath, overwrite=True)

                # Move dose.
                if create_moved_dose and moving_study.has_dose:
                    moving_dose = moving_study.dose_data
                    moved_dose = resample(moving_dose, offset=moving_study.ct_offset, output_offset=fixed_study.ct_offset, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study.id, p, moving_study.id, 'dose', f'{model}.nii.gz')
                    save_nifti(moved_dose, filepath, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', f'{model}.csv')
        timer.save(filepath)
