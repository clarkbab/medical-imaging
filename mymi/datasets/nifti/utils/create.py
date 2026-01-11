import os

from mymi.transforms import sitk_save_transform
from mymi.typing import *
from mymi.utils import *

from ..dataset import NiftiDataset
from ..series import NiftiModality

def create_ct(
    dataset: DatasetID,
    pat: PatientID,
    study: StudyID,
    series: NiftiSeriesID,
    data: CtImageArray,
    spacing: Spacing3D,
    origin: Point3D,
    dry_run: bool = True) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'patients', pat_id, study, 'ct', f'{series}.nii.gz')
    with_makeitso(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, origin=origin), f"Creating CT at {filepath}.")

def create_region(
    dataset: DatasetID,
    pat: PatientID,
    study: StudyID,
    series: NiftiSeriesID,
    region: RegionID,
    data: LabelArray,
    spacing: Spacing3D,
    origin: Point3D,
    dry_run: bool = True) -> None:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'patients', pat_id, study, 'regions', series, f'{region}.nii.gz')
    with_makeitso(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, origin=origin), f"Creating region at {filepath}.")

def create_registration_moved_image(
    dataset: DatasetID,
    fixed_pat: PatientID,
    model: ModelID,
    data: Union[CtImageArray, DoseImageArray, MrImageArray],
    modality: NiftiModality,
    spacing: Spacing3D,
    origin: Point3D,
    dry_run: bool = True,
    fixed_study: StudyID = 'study_1',
    moving_pat: Optional[PatientID] = None,
    moving_study: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset)
    moving_pat = fixed_pat if moving_pat is None else moving_pat
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat, fixed_study, moving_pat, moving_study, modality, f'{model}.nii.gz')
    with_makeitso(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, origin=origin), f"Creating moved image at {filepath}.")

def create_registration_moved_landmarks(
    dataset: DatasetID,
    fixed_pat: PatientID,
    model: ModelID,
    data: LandmarksFrame,
    dry_run: bool = True,
    fixed_study: StudyID = 'study_1',
    moving_pat: Optional[PatientID] = None,
    moving_study: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset)
    moving_pat = fixed_pat if moving_pat is None else moving_pat
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat, fixed_study, moving_pat, moving_study, 'landmarks', f'{model}.csv')
    with_makeitso(dry_run, lambda: save_csv(data, filepath), f"Creating moved landmarks at {filepath}.")

def create_registration_moved_region(
    dataset: DatasetID,
    fixed_pat: PatientID,
    region: RegionID,
    model: ModelID,
    data: LabelArray,
    spacing: Spacing3D,
    origin: Point3D,
    dry_run: bool = True,
    fixed_study: StudyID = 'study_1',
    moving_pat: Optional[PatientID] = None,
    moving_study: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset)
    moving_pat = fixed_pat if moving_pat is None else moving_pat
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat, fixed_study, moving_pat, moving_study, 'regions', region, f'{model}.nii.gz')
    with_makeitso(dry_run, lambda: save_nifti(data, filepath, spacing=spacing, origin=origin), f"Creating moved region at {filepath}.")

def create_registration_transform(
    dataset: DatasetID,
    fixed_pat: PatientID,
    model: ModelID,
    transform: sitk.Transform,
    dry_run: bool = True,
    fixed_study: StudyID = 'study_1',
    moving_pat: Optional[PatientID] = None,
    moving_study: StudyID = 'study_0') -> None:
    set = NiftiDataset(dataset)
    moving_pat = fixed_pat if moving_pat is None else moving_pat
    filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', fixed_pat, fixed_study, moving_pat, moving_study, 'transform', f'{model}.hdf5')
    with_makeitso(dry_run, lambda: sitk_save_transform(transform, filepath), f"Creating registration transform at {filepath}.")
