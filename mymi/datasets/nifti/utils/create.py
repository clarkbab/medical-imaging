import os

from mymi.transforms import sitk_save_transform
from mymi.typing import *
from mymi.utils import *

from ..dataset import NiftiDataset

def create_ct(
    dataset_id: DatasetID,
    pat_id: PatientID,
    study_id: StudyID,
    series_id: NiftiSeriesID,
    data: CtVolume,
    spacing: Spacing3D,
    offset: Point3D,
    dry_run: bool = True) -> None:
    set = NiftiDataset(dataset_id)
    filepath = os.path.join(set.path, 'data', 'patients', pat_id, study_id, 'ct', f'{series_id}.nii.gz')
    with_dry_run(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, offset=offset), f"Creating CT at {filepath}.")

def create_region(
    dataset_id: DatasetID,
    pat_id: PatientID,
    study_id: StudyID,
    series_id: NiftiSeriesID,
    region_id: RegionID,
    data: LabelData3D,
    spacing: Spacing3D,
    offset: Point3D,
    dry_run: bool = True) -> None:
    set = NiftiDataset(dataset_id)
    filepath = os.path.join(set.path, 'data', 'patients', pat_id, study_id, 'regions', series_id, f'{region_id}.nii.gz')
    with_dry_run(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, offset=offset), f"Creating region at {filepath}.")

def create_registration_moved_image(
    dataset_id: DatasetID,
    fixed_pat_id: PatientID,
    model: ModelID,
    data: Union[CtVolume, DoseData, MrData],
    modality: NiftiModality,
    spacing: Spacing3D,
    offset: Point3D,
    dry_run: bool = True,
    fixed_study_id: StudyID = 'study_1',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset_id)
    moving_pat_id = fixed_pat_id if moving_pat_id is None else moving_pat_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, modality, f'{model}.nii.gz')
    with_dry_run(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, offset=offset), f"Creating moved image at {filepath}.")

def create_registration_moved_landmarks(
    dataset_id: DatasetID,
    fixed_pat_id: PatientID,
    model: ModelID,
    data: LandmarksData,
    dry_run: bool = True,
    fixed_study_id: StudyID = 'study_1',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset_id)
    moving_pat_id = fixed_pat_id if moving_pat_id is None else moving_pat_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'landmarks', f'{model}.csv')
    with_dry_run(dry_run, lambda: save_csv(data, filepath), f"Creating moved landmarks at {filepath}.")

def create_registration_moved_region(
    dataset_id: DatasetID,
    fixed_pat_id: PatientID,
    region_id: RegionID,
    model: ModelID,
    data: LabelData,
    spacing: Spacing3D,
    offset: Point3D,
    dry_run: bool = True,
    fixed_study_id: StudyID = 'study_1',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset_id)
    moving_pat_id = fixed_pat_id if moving_pat_id is None else moving_pat_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'regions', region_id, f'{model}.nii.gz')
    with_dry_run(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, offset=offset), f"Creating moved region at {filepath}.")

def create_registration_transform(
    dataset_id: DatasetID,
    fixed_pat_id: PatientID,
    model: ModelID,
    transform: sitk.Transform,
    dry_run: bool = True,
    fixed_study_id: StudyID = 'study_1',
    moving_pat_id: Optional[PatientID] = None,
    moving_study_id: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset_id)
    moving_pat_id = fixed_pat_id if moving_pat_id is None else moving_pat_id
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id, 'dvf', f'{model}.hdf5')
    with_dry_run(dry_run, lambda: sitk_save_transform(transform, filepath), f"Creating registration transform at {filepath}.")
