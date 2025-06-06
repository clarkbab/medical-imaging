import itk
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
import torch
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi.models import load_model
from mymi.models.architectures import RegMod
from mymi.regions import regions_to_list
from mymi.transforms import crop, resample, load_sitk_transform, save_sitk_transform, sitk_transform_image, sitk_transform_points
from mymi.typing import *
from mymi.utils import *

def create_patient_registration(
    dataset: str,
    fixed_pat_id: PatientID,
    module: torch.nn.Module,
    project: str,
    model_name: str,
    model_spacing: Spacing3D,
    device: Optional[torch.device] = None,
    fixed_study_id: StudyID = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0',
    pad_threshold: Optional[float] = -1024,
    replace_padding: Optional[float] = -1024,
    regions: Optional[Regions] = 'all',
    regions_ignore_missing: bool = True) -> None:
    # Load model.
    model, device, ckpt_info = load_model(module, project, model_name, ckpt='best', state='eval')
    norm_params = ckpt_info['norm-params']

    # Load data.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })
    fixed_pat = set.patient(fixed_pat_id)
    moving_pat = set.patient(moving_pat_id) if moving_pat_id is not None else fixed_pat
    fixed_study = fixed_pat.study(fixed_study_id)
    moving_study = moving_pat.study(moving_study_id)
    fixed_ct = fixed_study.ct_data
    fixed_size = fixed_ct.shape
    fixed_spacing = fixed_study.ct_spacing
    fixed_offset = fixed_study.ct_offset
    moving_ct = moving_study.ct_data
    moving_spacing = moving_study.ct_spacing
    moving_offset = moving_study.ct_offset
    assert moving_spacing == fixed_spacing, "Fixed and moving images should have same spacing - initial rigid registration."

    # Replace padding before passing through model.
    if pad_threshold is not None and replace_padding is not None:
        fixed_ct[fixed_ct < pad_threshold] = replace_padding
        moving_ct[moving_ct < pad_threshold] = replace_padding

    # Resample to model spacing.
    input = np.stack([fixed_ct, moving_ct])
    input = resample(input, output_spacing=model_spacing, spacing=fixed_spacing)

    # Normalise input.
    input = (input - norm_params['mean']) / norm_params['std']

    # Perform inference.
    input = np.expand_dims(input, 0).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    input = torch.tensor(input, device=device)
    _, dvf = model(input)

    # # Perform post-processing.
    dvf = dvf.squeeze(0)  # Remove batch and CT channel dims.
    dvf = dvf.detach().cpu().numpy()
    dvf = resample(dvf, output_spacing=fixed_spacing, spacing=model_spacing)
    crop_box = ((0, 0, 0), fixed_size)
    dvf = crop(dvf, crop_box)   # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.

    # Convert DVF to sitk format - each voxel represents displacement in patient coordinates.
    dvf = dvf / 2    # Network operates on [-1, 1) scale for DVF values. 
    dvf = np.moveaxis(dvf, 0, -1) * fixed_spacing * fixed_size + fixed_offset
    dvf = np.moveaxis(dvf, -1, 0)

    # Don't use y_moved predicted from network. Otherwise CT moved will have been resampled 3 times
    # Use resampled DVF to transform moving -> fixed CT using one resample. Otherwise, CT will have
    # been resampled 3 times: moving (original) -> moving (model) -> moved (model) -> moved(original).
    sitk_transform = dvf_to_sitk_transform(dvf, fixed_spacing, fixed_offset)
    ct_moved = sitk_transform_image(moving_ct, sitk_transform, fixed_ct.shape, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing)
    ct_moved = crop(ct_moved, crop_box)   # Resampling rounds *up* to nearest number of voxels, cropping may be necessary to obtain original image size.

    # Save moved CT.
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'ct', f'{model_name}.nii.gz')
    save_nifti(ct_moved, filepath, spacing=fixed_spacing, offset=fixed_offset)

    # Save DVF.
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'dvf', f'{model_name}.hdf5')
    save_sitk_transform(sitk_transform, filepath)

    # Move region data.
    if regions is not None:
        moving_region_data = moving_study.region_data(regions=regions, regions_ignore_missing=regions_ignore_missing)
        if moving_region_data is not None:
            for region, moving_label in moving_region_data.items():
                # Apply registration transform.
                moved_label = sitk_transform_image(moving_label, sitk_transform, fixed_ct.shape, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing)
                moved_label = crop(moved_label, crop_box)
                filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'regions', region, f'{model_name}.nii.gz')
                save_nifti(moved_label, filepath, spacing=fixed_spacing, offset=fixed_offset)

    # Move landmarks.
    if landmarks is not None:
        fixed_landmark_data = fixed_study.landmark_data(landmarks=landmarks)
        if fixed_landmark_data is not None:
            # Move landmarks from fixed -> moving spacing - we can't always invert DVF transforms.
            fixed_points = fixed_landmark_data[list(range(3))].to_numpy()
            moved_points = sitk_transform_points(fixed_points, sitk_transform)
            moved_landmark_data = fixed_landmark_data.copy()
            moved_landmark_data[list(range(3))] = moved_points
            landmark_cols = ['landmark-id', 0, 1, 2]    # Don't save patient-id/study-id cols.
            moved_landmark_data = moved_landmark_data[landmark_cols]
            filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'landmarks', f'{model_name}.csv')
            save_csv(moved_landmark_data, filepath)

def delete_patient_registration(
    dataset: str,
    fixed_pat_id: PatientID,
    model_name: str,
    fixed_study_id: StudyID = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0',
    regions: Optional[Regions] = 'all') -> None:

    # Remove moved CT.
    set = NiftiDataset(dataset)
    fixed_pat = set.patient(fixed_pat_id)
    fixed_study = fixed_pat.study(fixed_study_id)
    moving_pat_id = fixed_pat_id if moving_pat_id is None else moving_pat_id
    moving_pat = set.patient(moving_pat_id)
    moving_study = moving_pat.study(moving_study_id)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'ct', f'{model_name}.nii.gz')
    if os.path.exists(filepath):
        os.remove(filepath)

    # Remove transform.
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'dvf', f'{model_name}.hdf5')
    if os.path.exists(filepath):
        os.remove(filepath)

    # Remove region data.
    if regions is not None:
        moving_regions = moving_study.list_regions(regions=regions)
        for r in moving_regions:
            filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'regions', r, f'{model_name}.nii.gz')
            if os.path.exists(filepath):
                os.remove(filepath)

    # Move landmarks.
    if landmarks is not None:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'landmarks', f'{model_name}.csv')
        if os.path.exists(filepath):
            os.remove(filepath)

def create_patient_identity_registration(
    dataset: str,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    model_name: str = 'identity',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0',
    regions: Optional[Regions] = 'all',
    regions_ignore_missing: bool = True) -> None:

    # Load data.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })
    fixed_pat = set.patient(fixed_pat_id)
    moving_pat = set.patient(moving_pat_id) if moving_pat_id is not None else fixed_pat
    fixed_study = fixed_pat.study(fixed_study_id)
    moving_study = moving_pat.study(moving_study_id)
    fixed_ct = fixed_study.ct_data
    fixed_size = fixed_ct.shape
    fixed_spacing = fixed_study.ct_spacing
    moving_ct = moving_study.ct_data
    moving_spacing = moving_study.ct_spacing
    # assert moving_spacing == fixed_spacing, "Fixed and moving images should have same spacing - initial rigid registration."

    # Resample moving to moved.
    # Don't use stored offsets, these could be very different between fixed and moving images. Just align image centres.
    fixed_offset = tuple(-np.array(fixed_study.ct_fov) / 2)
    moving_offset = tuple(-np.array(moving_study.ct_fov) / 2)
    moved_ct = resample(moving_ct, offset=moving_offset, output_offset=fixed_offset, output_size=fixed_size, output_spacing=fixed_spacing, spacing=moving_spacing)

    # Save moved CT.
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'ct', f'{model_name}.nii.gz')
    save_nifti(moved_ct, filepath, spacing=fixed_spacing, offset=fixed_offset)

    # Save transform.
    transform = sitk.Transform(3, sitk.sitkIdentity)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'dvf', f'{model_name}.hdf5')
    save_sitk_transform(transform, filepath)

    # Move region data.
    if regions is not None:
        moving_region_data = moving_study.region_data(regions=regions, regions_ignore_missing=regions_ignore_missing)
        if moving_region_data is not None:
            for region, moving_label in moving_region_data.items():
                # Apply registration transform.
                moved_label = resample(moving_label, offset=moving_offset, output_offset=fixed_offset, output_size=fixed_size, output_spacing=fixed_spacing, spacing=moving_spacing)
                filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'regions', region, f'{model_name}.nii.gz')
                save_nifti(moved_label, filepath, spacing=fixed_spacing, offset=fixed_offset)

    # Move landmarks.
    if landmarks is not None:
        fixed_landmark_data = fixed_study.landmark_data(landmarks=landmarks)
        if fixed_landmark_data is not None:
            # Use fixed landmarks as moved landmarks.
            fixed_points = fixed_landmark_data[list(range(3))].to_numpy()
            moved_landmark_data = fixed_landmark_data.copy()
            moved_landmark_data[list(range(3))] = fixed_points
            landmark_cols = ['landmark-id', 0, 1, 2]    # Don't save patient-id/study-id cols.
            moved_landmark_data = moved_landmark_data[landmark_cols]
            filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat.id, fixed_study.id, moving_pat.id, moving_study.id, 'landmarks', f'{model_name}.csv')
            save_csv(moved_landmark_data, filepath)

def create_registrations(
    dataset: str,
    project: str,
    model: str,
    model_spacing: Spacing3D,
    module: torch.nn.Module = RegMod,
    pat_ids: PatientIDs = 'all',
    splits: Splits = 'all',
    use_timing: bool = True,
    **kwargs) -> None:

    # Create timing table.
    if use_timing:
        cols = {
            'dataset': str,
            'patient-id': str,
        }
        timer = Timer(cols)

    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    # Make predictions.
    for p in tqdm(pat_ids):
        # Timing table data.
        data = {
            'dataset': dataset,
            'patient-id': p,
        }

        with timer.record(data, enabled=use_timing):
            if model == 'identity':
                create_patient_identity_registration(dataset, p, **kwargs)
            else:
                create_patient_registration(dataset, p, module, project, model, model_spacing, **kwargs)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        timer.save(filepath)

def delete_registrations(
    dataset: str,
    model: str,
    splits: Splits = 'all',
    **kwargs) -> None:
    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(splits=splits)

    # Remove predictions.
    for p in tqdm(pat_ids):
        delete_patient_registration(dataset, p, model, **kwargs)

def load_registration(
    dataset: str,
    fixed_pat_id: PatientID,
    model: str,
    fixed_study_id: StudyID = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0',
    raise_error: bool = True,
    regions: Optional[Regions] = 'all',
    use_patient_coords: bool = True) -> Tuple[CtImage, Union[itk.Transform, sitk.Transform], Optional[RegionData], Optional[Landmarks]]:
    # Load moved CT.
    if moving_pat_id is None:
        moving_pat_id = fixed_pat_id
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'ct', f'{model}.nii.gz')
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"CT registration not found at '{filepath}'.")
        else:
            return None
    moved_ct, _, _ = load_nifti(filepath)

    # Load transform.
    suffixes = ['.hdf5', '.nii', '.nii.gz']
    transform = None
    base_path = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'dvf')
    for s in suffixes:
        filepath = os.path.join(base_path, f'{model}{s}')
        if os.path.exists(filepath):
            transform = load_sitk_transform(filepath)
    if transform is None:
        raise ValueError(f"Transform not found for model '{model}' at '{base_path}'. Allowed suffixes: {suffixes}.")

    if regions is not None:
        # Load moved regions.
        moving_study = set.patient(moving_pat_id).study(moving_study_id)
        regions = regions_to_list(regions, literals={ 'all': moving_study.list_regions })
        moved_region_data = {}
        for r in regions:
            rdata = load_registered_region(dataset, fixed_pat_id, model, r, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id)
            if rdata is None:
                continue
            moved_region_data[r] = rdata
    else:
        moved_region_data = None

    if landmarks is not None:
        # Load landmarks - moved from fixed to moving space (reversed).
        moved_landmark_data = load_registered_landmarks(dataset, fixed_pat_id, model, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, use_patient_coords=use_patient_coords)
        if moved_landmark_data is not None:
            all_landmarks = list(sorted(moved_landmark_data['landmark-id'].unique()))

            # Filter based on requested landmarks.
            if isinstance(landmarks, float) and landmarks > 0 and landmarks < 1:
                landmarks = p_landmarks(all_landmarks, landmarks)
            else:
                landmarks = arg_to_list(landmarks, Landmark, literals={ 'all': all_landmarks })
            moved_landmark_data = moved_landmark_data[moved_landmark_data['landmark-id'].isin(landmarks)]
    else:
        moved_landmark_data = None
            
    return moved_ct, transform, moved_region_data, moved_landmark_data

def load_registered_region(
    dataset: str,
    fixed_pat_id: PatientID,
    model: str,
    region: Region,
    fixed_study_id: StudyID = 'study_1',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0') -> Optional[RegionLabel]:
    moving_pat_id = moving_pat_id if moving_pat_id is not None else fixed_pat_id
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'regions', region, f'{model}.nii.gz')
    if not os.path.exists(filepath):
        return None
    data, _, _ = load_nifti(filepath)
    data = data.astype(np.bool_)
    return data

# These are registered to moving space!
def load_registered_landmarks(
    dataset: str,
    fixed_pat_id: PatientID,
    model: str,
    fixed_study_id: StudyID = 'study_1',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0',
    use_patient_coords: bool = True) -> Optional[Landmarks]:
    moving_pat_id = moving_pat_id if moving_pat_id is not None else fixed_pat_id

    # Load landmarks.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'landmarks', f'{model}.csv')
    if not os.path.exists(filepath):
        return None
    map_cols = dict((str(i), i) for i in range(3))
    landmarks = load_files_csv(filepath, map_cols=map_cols)

    # Convert to image coordinates.
    if not use_patient_coords:
        study = set.patient(moving_pat_id).study(moving_study_id)
        spacing = study.ct_spacing
        offset = study.ct_offset
        lm_data = landmarks[list(range(3))].to_numpy()
        lm_data = (lm_data - offset) / spacing
        lm_data = lm_data.round()
        lm_data = lm_data.astype(np.uint32)
        landmarks[list(range(3))] = lm_data

    # Add patient information - to handle concatenation easily with other patient/study landmarks.
    if 'patient-id' not in landmarks.columns:
        landmarks.insert(0, 'patient-id', moving_pat_id)
    if 'study-id' not in landmarks.columns:
        landmarks.insert(1, 'study-id', moving_study_id)

    return landmarks
