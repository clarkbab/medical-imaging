import SimpleITK as sitk
from typing import *

from mymi.datasets import NiftiDataset
from mymi.typing import *

from ..registration import rigid_image_registration
from ..resample import resample
from ..sitk import sitk_transform_points

def rigid_registration(
    dataset: str,
    moving_pat: PatientID,
    moving_study: StudyID,
    fixed_pat: PatientID,
    fixed_study: StudyID,
    landmarks: Optional[LandmarkIDs] = None,
    regions: Optional[Regions] = None,
    regions_ignore_missing: bool = False,
    **kwargs) -> Tuple[CtImageArray, Optional[RegionArrays], Optional[LandmarksFrame], sitk.Transform]:

    # Load CT data.
    set = NiftiDataset(dataset)
    moving_pat = set.patient(moving_pat_id)
    moving_study = moving_pat.study(moving_study)
    moving_ct = moving_study.ct_data
    moving_spacing = moving_study.ct_spacing
    moving_origin = moving_study.ct_origin
    fixed_pat = set.patient(fixed_pat_id)
    fixed_study = fixed_pat.study(fixed_study)
    fixed_ct = fixed_study.ct_data
    fixed_spacing = fixed_study.ct_spacing
    fixed_origin = fixed_study.ct_origin

    # Do we need to resample before applying rigid registration?
    # The rigid registration transform operates on patient coordinates, and the object's patient
    # coordinates are not affected by resampling.
    # moving_ct = resample(moving_ct, spacing=moving_spacing, output_spacing=fixed_spacing)

    # Perform CT registration.
    moved_ct, transform = rigid_image_registration(moving_ct, moving_spacing, moving_origin, fixed_ct, fixed_spacing, fixed_origin, **kwargs)

    # Move region data.
    moved_regions_data = None
    if regions is not None:
        moving_regions_data = moving_study.regions_data(regions=regions, regions_ignore_missing=regions_ignore_missing)
        if moving_regions_data is not None:
            moved_regions_data = {}
            for region, moving_label in moving_regions_data.items():
                # Apply registration transform.
                moved_label = resample(moving_label, origin=moving_origin, output_origin=fixed_origin, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                moved_regions_data[region] = moved_label

    # Move landmarks.
    moved_landmarks_data = None
    if landmarks is not None:
        moving_landmarks_data = moving_study.landmarks_data(landmarks=landmarks)
        if moving_landmarks_data is not None:
            moving_points = moving_landmarks_data[list(range(3))].to_numpy()
            moved_landmarks_data = moving_landmarks_data.copy()
            # It's rigid, easy inverse.
            inv_transform = transform.GetInverse()
            moved_points = sitk_transform_points(moving_points, inv_transform)
            moved_landmarks_data[list(range(3))] = moved_points

    return moved_ct, moved_regions_data, moved_landmarks_data, transform
