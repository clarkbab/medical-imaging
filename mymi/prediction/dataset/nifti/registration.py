import itk
import numpy as np
import os
import pandas as pd
import SimpleITK as sitk
from typing import Literal, Optional, Tuple, Union

from mymi.dataset import NiftiDataset
from mymi.regions import regions_to_list
from mymi.transforms import itk_load_transform, sitk_load_transform
from mymi.types import CtImage, Landmarks, PatientLandmarks, PatientID, PatientRegion, PatientRegions, RegionImage, RegionImages, StudyID
from mymi.utils import arg_to_list, load_nifti

def load_registration(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    model: str,
    landmarks: Optional[PatientLandmarks] = None,
    regions: Optional[PatientRegions] = None,
    regions_ignore_missing: bool = False,
    transform_format: Literal['itk', 'sitk'] = 'sitk',
    use_image_coords: bool = False) -> Tuple[CtImage, Union[itk.Transform, sitk.Transform], Optional[RegionImages], Optional[Landmarks]]:
    # Load moved CT.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, 'ct', 'series_0.nii.gz')
    if not os.path.exists(filepath):
        raise ValueError(f"CT registration not found at '{filepath}'.")
    moved_ct, _, _ = load_nifti(filepath)

    # Load transform.
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, 'dvf', 'series_0.hdf5')
    if transform_format == 'itk':
        transform = itk_load_transform(filepath)
    elif transform_format == 'sitk':
        transform = sitk_load_transform(filepath)
    else:
        raise ValueError(f"Unrecognised transform_format '{transform_format}'.")

    # Load moved regions.
    if regions is not None:
        moving_study = set.patient(moving_pat_id).study(moving_study_id)
        regions = regions_to_list(regions, literals={ 'all': moving_study.list_regions })
        moved_region_data = {}
        for r in regions:
            rdata = load_registered_region(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, r, raise_error=not regions_ignore_missing)
            if rdata is None:
                continue
            moved_region_data[r] = rdata
    else:
        moved_region_data = None

    # Load landmarks - in moved space.
    if landmarks is not None:
        # All landmarks are stored in a single file.
        moved_landmarks = load_registered_landmarks(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, raise_error=not regions_ignore_missing, use_image_coords=use_image_coords)
        # Filter based on requested landmarks.
        landmarks = regions_to_list(landmarks, literals={ 'all': set.list_landmarks })
        moved_landmarks = moved_landmarks[moved_landmarks['landmark-id'].isin(landmarks)]
    else:
        moved_landmarks = None
            
    return moved_ct, transform, moved_region_data, moved_landmarks

def load_registered_region(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    model: str,
    region: PatientRegion,
    raise_error: bool = True) -> Optional[RegionImage]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, 'regions', 'series_1', f'{region}.nii.gz')
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Registered region '{region}' not found at '{filepath}'.")
        else:
            return None
    data, _, _ = load_nifti(filepath)
    data = data.astype(np.bool_)
    return data

# These are registered to moving space!
def load_registered_landmarks(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    model: str,
    raise_error: bool = True,
    use_image_coords: bool = False) -> Optional[RegionImage]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, 'landmarks', 'series_1.csv')
    if not os.path.exists(filepath):
        if raise_error:
            raise ValueError(f"Registered landmarks not found at '{filepath}'.")
        else:
            return None
    landmarks = pd.read_csv(filepath)
    landmarks = landmarks.rename(columns={ '0': 0, '1': 1, '2': 2 })

    # Convert to image coordinates.
    if use_image_coords:
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
