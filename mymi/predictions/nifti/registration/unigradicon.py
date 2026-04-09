import numpy as np
import os
import SimpleITK as sitk
import subprocess
import torch
from tqdm import tqdm
from typing import *
import unigradicon as ugi

from dicomset.nifti import NiftiDataset
from dicomset.utils import create_affine
from mymi import logging
from mymi.predictions.registration import register_unigradicon
from mymi.regions import regions_to_list
from mymi.transforms import load_itk_transform, sitk_load_transform, resample, sitk_save_transform, sitk_transform_points
from mymi.typing import *
from mymi.utils.io import save_csv
from mymi.utils.nifti import save_nifti
from mymi.utils.timer import Timer

def load_finetuned_model(
    dataset: DatasetID,
    run_name: str,
    ckpt: str) -> torch.nn.Module:
    model = ugi.get_unigradicon()
    filepath = os.path.join('models', 'unigradicon', dataset, run_name, ckpt) 
    trained_weights = torch.load(filepath)
    model.regis_net.load_state_dict(trained_weights)
    model.to(icon.config.device)
    return model

def create_unigradicon_finetuned_predictions(
    dataset: DatasetID,
    run_name: str,
    ckpt: str,
    model: str,
    create_moved_dose: bool = True,
    create_moved_ct: bool = True,
    landmarks: Optional[LandmarkIDs] = 'all',
    pat_ids: PatientIDs = 'all',
    regions: Optional[RegionIDs] = 'all',
    splits: Optional[Split] = 'all',
    use_io: bool = False,
    use_timing: bool = True) -> None:
    logging.arg_log('Making UniGradICON (fine-tuned) predictions', ('dataset', 'model', 'regions'), (dataset, model, regions))

    # Create timing table.
    if use_timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'device': str
        }
        timer = Timer(cols)

    # Load model.
    model = load_finetuned_model(dataset, run_name, ckpt)

    # Load patients.
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
            # Load images.
            moving_pat_id, moving_study = p, 'study_0'
            fixed_pat_id, fixed_study = p, 'study_1'
            moving_pat = set.patient(moving_pat_id)
            fixed_pat = set.patient(fixed_pat_id)
            moving_study = moving_pat.study(moving_study)
            fixed_study = fixed_pat.study(fixed_study)
            moving_ct = moving_study.ct_data
            fixed_ct = fixed_study.ct_data

            # Process images for network input.
            min_val, max_val = -1000, 1000
            moving, fixed = moving.clamp(min_val, max_val), fixed.clamp(min_val, max_val)
            moving, fixed = (moving - min_val) / (max_val - min_val), (fixed - min_val) / (max_val - min_val)
            moving, fixed = moving.float().cuda(), fixed.float().cuda()

    # fixed = itk.imread(args.fixed)
    # moving = itk.imread(args.moving)

    # if args.io_iterations == "None":
    #     io_iterations = None
    # else:
    #     io_iterations = int(args.io_iterations)

    # phi_AB, phi_BA = icon_registration.itk_wrapper.register_pair(
    #     net,
    #     preprocess(moving),
    #     preprocess(fixed),
    #     finetune_steps=io_iterations)

    # itk.transformwrite([phi_AB], args.transform_out)

    # if args.warped_moving_out:
    #     moving = itk.CastImageFilter[type(moving), itk.Image[itk.F, 3]].New()(moving)
    #     interpolator = itk.LinearInterpolateImageFunction.New(moving)
    #     warped_moving_image = itk.resample_image_filter(
    #             moving,
    #             transform=phi_AB,
    #             interpolator=interpolator,
    #             use_reference_image=True,
    #             reference_image=fixed
    #             )
    #     itk.imwrite(warped_moving_image, args.warped_moving_out)

            # Register CT images.
            if create_moved_ct:
                os.makedirs(os.path.dirname(moved_path), exist_ok=True)
                os.makedirs(os.path.dirname(transform_path), exist_ok=True)
                io_iterations = 50 if use_io else None
                command = [
                    'unigradicon-register',
                    '--moving', moving_study.ct_filepath,
                    '--moving_modality', 'ct',
                    '--fixed', fixed_study.ct_filepath,
                    '--fixed_modality', 'ct',
                    '--warped_moving_out', moved_path,
                    '--transform_out', transform_path,
                    '--io_iterations', str(io_iterations)
                ]
                logging.info(command)
                subprocess.run(command)

                # Convert transform to sitk.
                t_itk = load_itk_transform(transform_path)[0]
                transform = convert_transform_to_sitk(t_itk)
                sitk_save_transform(transform, transform_path)

                # Apply transform to moving CT.
                fixed_ct = fixed_study.ct_data
                fixed_spacing = fixed_study.ct_spacing
                fixed_origin = fixed_study.ct_origin
                moving_ct = moving_study.ct_data
                moving_spacing = moving_study.ct_spacing
                moving_origin = moving_study.ct_origin
                moved_ct = resample(moving_ct, origin=moving_origin, output_origin=fixed_origin, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                save_nifti(moved_ct, moved_path, spacing=fixed_spacing, origin=fixed_origin)

            # Register regions.
            if regions is not None:
                regions = regions_to_list(regions, literals={ 'all': set.list_regions })
                for r in regions:
                    if not moving_study.has_region(r):
                        continue

                    # Load data.
                    fixed_ct = fixed_study.ct_data
                    fixed_spacing = fixed_study.ct_spacing
                    fixed_origin = fixed_study.ct_origin
                    moving_label = moving_study.regions_data(regions=r)[r]
                    moving_spacing = moving_study.ct_spacing
                    moving_origin = moving_study.ct_origin
                    transform = sitk_load_transform(transform_path)

                    # Perform transform.
                    moved_label = resample(moving_label, origin=moving_origin, output_origin=fixed_origin, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                    moved_label_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                    os.makedirs(os.path.dirname(moved_label_path), exist_ok=True)
                    save_nifti(moved_label, moved_label_path, spacing=fixed_spacing, origin=fixed_origin)

                    # # Perform region warp.
                    # moving_region_path = moving_study.region_path(r)
                    # moved_region_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                    # os.makedirs(os.path.dirname(moved_region_path), exist_ok=True)
                    # command = [
                    #     'unigradicon-warp',
                    #     '--moving', moving_region_path,
                    #     '--fixed', fixed_study.ct_path,
                    #     '--transform', transform_path,
                    #     '--warped_moving_out', moved_region_path,
                    #     '--nearest_neighbor'
                    # ]
                    # logging.info(command)
                    # subprocess.run(command)

            # Transform any fixed landmarks back to moving space.
            if landmarks is not None:
                transform = sitk_load_transform(transform_path)
                if fixed_study.has_landmark(landmarks=landmarks):
                    fixed_lm_df = fixed_study.landmarks_data(landmarks=landmarks)
                    lm_data = fixed_lm_df[list(range(3))].to_numpy()
                    lm_data_t = sitk_transform_points(lm_data, transform)
                    if np.allclose(lm_data_t, lm_data):
                        logging.warning(f"Moved points are very similar to fixed points - identity transform?")
                    moving_lm_df = fixed_lm_df.copy()
                    moving_lm_df[list(range(3))] = lm_data_t

                    # Save transformed points.
                    filepath = os.path.join(reg_path, 'landmarks', f'{model}.csv')
                    save_csv(moving_lm_df, filepath, overwrite=True)
                    
            # Move dose.
            if create_moved_dose and moving_study.has_dose:
                moving_dose = moving_study.dose_data
                moved_dose = resample(moving_dose, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study.id, p, moving_study.id, 'dose', f'{model}.nii.gz')
                save_nifti(moved_dose, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
        timer.save(filepath)

def create_unigradicon_predictions(
    dataset: str,
    model: str,
    create_moved_dose: bool = True,
    create_moved_ct: bool = True,
    landmarks: Optional[LandmarkIDs] = 'all',
    pat_ids: PatientIDs = 'all',
    regions: Optional[RegionIDs] = 'all',
    splits: Optional[Split] = 'all',
    use_io: bool = False,
    use_timing: bool = True) -> None:
    logging.arg_log('Making UniGradICON predictions', ('dataset', 'model', 'regions'), (dataset, model, regions))

    # Create timing table.
    if use_timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patients.
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
            # Load details.
            moving_pat_id, moving_study = p, 'study_0'
            fixed_pat_id, fixed_study = p, 'study_1'
            moving_pat = set.patient(moving_pat_id)
            fixed_pat = set.patient(fixed_pat_id)
            moving_study = moving_pat.study(moving_study)
            fixed_study = fixed_pat.study(fixed_study)
            reg_path = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat_id, fixed_study.id, moving_pat_id, moving_study.id)

            # Prepare data.
            fixed_ct = fixed_study.ct_data
            moving_ct = moving_study.ct_data
            fixed_affine = create_affine(fixed_study.ct_spacing, fixed_study.ct_origin)
            moving_affine = create_affine(moving_study.ct_spacing, moving_study.ct_origin)

            # Core registration.
            transform = register_unigradicon(
                fixed_ct, moving_ct, fixed_affine, moving_affine,
                use_io=use_io)

            # Save transform.
            transform_path = os.path.join(reg_path, 'transform', f'{model}.hdf5')
            os.makedirs(os.path.dirname(transform_path), exist_ok=True)
            sitk_save_transform(transform, transform_path)

            # Save moved CT.
            if create_moved_ct:
                moved_ct = resample(moving_ct, affine=moving_affine, output_affine=fixed_affine, transform=transform)
                moved_path = os.path.join(reg_path, 'ct', f'{model}.nii.gz')
                os.makedirs(os.path.dirname(moved_path), exist_ok=True)
                save_nifti(moved_ct, moved_path, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

            # Register regions.
            if regions is not None:
                regions = regions_to_list(regions, literals={ 'all': set.list_regions })
                for r in regions:
                    if not moving_study.has_region(r):
                        continue

                    moving_label = moving_study.regions_data(regions=r)[r]
                    moved_label = resample(moving_label, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                    moved_label_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                    os.makedirs(os.path.dirname(moved_label_path), exist_ok=True)
                    save_nifti(moved_label, moved_label_path, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

                    # # Perform region warp.
                    # moving_region_path = moving_study.region_path(r)
                    # moved_region_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                    # os.makedirs(os.path.dirname(moved_region_path), exist_ok=True)
                    # command = [
                    #     'unigradicon-warp',
                    #     '--moving', moving_region_path,
                    #     '--fixed', fixed_study.ct_path,
                    #     '--transform', transform_path,
                    #     '--warped_moving_out', moved_region_path,
                    #     '--nearest_neighbor'
                    # ]
                    # logging.info(command)
                    # subprocess.run(command)

            # Transform any fixed landmarks back to moving space.
            if landmarks is not None:
                transform = sitk_load_transform(transform_path)
                if fixed_study.has_landmark(landmarks=landmarks):
                    fixed_lm_df = fixed_study.landmarks_data(landmarks=landmarks)
                    lm_data = fixed_lm_df[list(range(3))].to_numpy()
                    lm_data_t = sitk_transform_points(lm_data, transform)
                    if np.allclose(lm_data_t, lm_data):
                        logging.warning(f"Moved points are very similar to fixed points - identity transform?")
                    moving_lm_df = fixed_lm_df.copy()
                    moving_lm_df[list(range(3))] = lm_data_t

                    # Save transformed points.
                    filepath = os.path.join(reg_path, 'landmarks', f'{model}.csv')
                    save_csv(moving_lm_df, filepath, overwrite=True)
                    
            # Move dose.
            if create_moved_dose and moving_study.has_dose:
                moving_dose = moving_study.dose_data
                moved_dose = resample(moving_dose, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study.id, p, moving_study.id, 'dose', f'{model}.nii.gz')
                save_nifti(moved_dose, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
        timer.save(filepath)
