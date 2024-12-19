import SimpleITK as sitk
from typing import Optional, Tuple

from mymi.dataset import NiftiDataset
from mymi.types import CtImage, PatientID, PatientLandmarks, PatientRegions, RegionImages, StudyID

from ..registration import rigid_image_registration
from ..resample import resample
from ..sitk import sitk_transform_image, sitk_transform_points

def rigid_registration(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    landmarks: Optional[PatientLandmarks] = None,
    regions: Optional[PatientRegions] = None,
    regions_ignore_missing: bool = False) -> Tuple[CtImage, RegionImages, sitk.Transform]:

    # Load CT data.
    set = NiftiDataset(dataset)
    moving_pat = set.patient(moving_pat_id)
    moving_study = moving_pat.study(moving_study_id)
    moving_ct = moving_study.ct_data
    moving_spacing = moving_study.ct_spacing
    moving_offset = moving_study.ct_offset
    fixed_pat = set.patient(fixed_pat_id)
    fixed_study = fixed_pat.study(fixed_study_id)
    fixed_ct = fixed_study.ct_data
    fixed_spacing = fixed_study.ct_spacing
    fixed_offset = fixed_study.ct_offset

    # Do we need to resample before applying rigid registration?
    # The rigid registration transform operates on patient coordinates, and the object's patient
    # coordinates are not affected by resampling.
    # moving_ct = resample(moving_ct, spacing=moving_spacing, output_spacing=fixed_spacing)

    # Perform CT registration.
    moved_ct, transform = rigid_image_registration(moving_ct, moving_spacing, moving_offset, fixed_ct, fixed_spacing, fixed_offset)

    # Move region data.
    if regions is not None:
        moving_region_data = moving_study.region_data(regions=regions, regions_ignore_missing=regions_ignore_missing)
        moved_region_data = {}
        for region, moving_label in moving_region_data.items():
            # Apply registration transform.
            moved_label = sitk_transform_image(moving_label, moving_spacing, moving_offset, fixed_ct.shape, fixed_spacing, fixed_offset, transform)

            moved_region_data[region] = moved_label
    else:
        moved_region_data = None

    # Move landmarks.
    if landmarks is not None:
        moving_landmarks = moving_study.landmark_data(landmarks=landmarks)
        moving_points = moving_landmarks[list(range(3))].to_numpy()
        moved_landmarks = moving_landmarks.copy()
        # It's rigid, easy inverse.
        inv_transform = transform.GetInverse()
        moved_points = sitk_transform_points(moving_points, inv_transform)
        moved_landmarks[list(range(3))] = moved_points
    else:
        moved_landmarks = None

    return moved_ct, moved_region_data, moved_landmarks, transform