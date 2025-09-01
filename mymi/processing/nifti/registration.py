import os
import shutil
from tqdm import tqdm
from typing import *

from mymi.datasets.dicom import DicomDataset, create, exists, recreate as recreate_dicom
from mymi.datasets.nifti import NiftiDataset
from mymi import logging
from mymi.predictions.nifti import load_registration
from mymi.regions import regions_to_list
from mymi.transforms import sitk_save_transform
from mymi.transforms.dataset.nifti import rigid_registration
from mymi.typing import *
from mymi.utils import *
from ..processing import write_flag

def convert_registration_predictions_to_dicom(
    dataset: DatasetID,
    model: ModelID,
    convert_ct: bool = False,
    convert_dose: bool = False,
    convert_fixed: bool = False,
    convert_moved: bool = False,
    convert_moving: bool = False,
    dest_dataset: Optional[DatasetID] = None,
    dest_fixed_study_id: str = 'idx:1',
    fixed_study_id: str = 'study_1',
    landmark_ids: Optional[LandmarkIDs] = None,
    moving_study_id: str = 'study_0',
    pat_ids: PatientIDs = 'all',
    recreate: bool = False,
    region_ids: Optional[RegionIDs] = None,
    use_rtdose_template: bool = True)  -> None:
    # Get dest dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    if exists(dest_dataset):
        dest_set = recreate_dicom(dest_dataset) if recreate else DicomDataset(dest_dataset)
    else:
        dest_set = create(dest_dataset)
    base_path = os.path.join(dest_set.path, 'data', 'patients')
    
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids)
    for p in tqdm(pat_ids):
        # Write fixed/moving data.
        pat = set.patient(p)
        study_ids = [moving_study_id, fixed_study_id]
        converts = [convert_moving, convert_fixed]
        moving_ref_cts = None   # Required for 'moved' landmarks - in moving image space.
        for s, convert_study in zip(study_ids, converts):
            study = pat.study(s)

            # Write CTs.
            ref_cts = None  # Perhaps required for RTSTRUCT and RTDOSE.
            if study.has_ct:
                ct_dicoms = to_ct_dicoms(study.ct_data, study.ct_spacing, study.ct_offset, p, s)
                ref_cts = ct_dicoms
                if s == moving_study_id:
                    moving_ref_cts = ct_dicoms
                if convert_ct and convert_study:
                    for i, c in enumerate(ct_dicoms):
                        filepath = os.path.join(base_path, p, s, 'ct', 'series_0', f'{i:03}.dcm')
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        c.save_as(filepath)

            # Write RTSTRUCT (regions and landmarks).
            if convert_study and (landmark_ids is not None or region_ids is not None) and (study.has_landmark_data or study.has_region_data):
                rtstruct_dicom = to_rtstruct_dicom(ref_cts, region_data=study.region_data(region_ids=region_ids), landmark_data=study.landmark_data(landmark_ids=landmark_ids))
                filepath = os.path.join(base_path, p, s, 'rtstruct', 'series_1.dcm')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                rtstruct_dicom.save_as(filepath)

            # Write RTDOSE.
            if convert_study and convert_dose and study.has_dose:
                rtdose_dicom = to_rtdose_dicom(study.dose_data, study.dose_spacing, study.dose_offset, ref_ct=ref_cts[0])
                filepath = os.path.join(base_path, p, s, 'rtdose', 'series_2.dcm')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                rtdose_dicom.save_as(filepath)

        # Load moved data.
        _, moved_ct, moved_region_data, moved_landmark_data, moved_dose = load_registration(dataset, p, model, fixed_study_id=fixed_study_id, landmark_ids=landmark_ids, moving_study_id=moving_study_id, region_ids=region_ids)

        # Write moved CT.
        fixed_study = pat.study(fixed_study_id)
        moved_ref_cts = None
        if convert_ct and convert_moved and moved_ct is not None:
            moved_ref_cts = to_ct_dicoms(moved_ct, fixed_study.ct_spacing, fixed_study.ct_offset, p, model)
            for i, c in enumerate(moved_ref_cts):
                filepath = os.path.join(base_path, p, model, 'ct', 'series_0', f'{i:03}.dcm')
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                if convert_moved or not os.path.exists(filepath):
                    c.save_as(filepath)
        
        # Write moved landmarks to moving study.
        if convert_moved and moved_landmark_data is not None:
            rtstruct_dicom = to_rtstruct_dicom(moving_ref_cts, landmark_data=moved_landmark_data)
            filepath = os.path.join(base_path, p, moving_study_id, 'rtstruct', f'{model}.dcm')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if convert_moved or not os.path.exists(filepath):
                rtstruct_dicom.save_as(filepath)

        # Write moved RTSTRUCT (regions only, landmarks belong to moving study).
        if convert_moved and moved_region_data is not None:
            rtstruct_dicom = to_rtstruct_dicom(moved_ref_cts, region_data=moved_region_data)
            filepath = os.path.join(base_path, p, model, 'rtstruct', 'series_1.dcm')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if convert_moved or not os.path.exists(filepath):
                rtstruct_dicom.save_as(filepath)

        # Write moved RTDOSE.
        if convert_moved and moved_dose is not None:
            if use_rtdose_template:
                # Use an existing rtdose file from the dest dataset.
                dest_pat = dest_set.patient(p)
                dest_fixed_study = dest_pat.study(dest_fixed_study_id)
                if dest_fixed_study.has_rtdose:
                    rtdose_template = dest_fixed_study.default_rtdose.dicom
                    # Dose has been resample to the fixed CT spacing when transform was applied.
                    rtdose_dicom = to_rtdose_dicom(moved_dose, fixed_study.ct_spacing, fixed_study.ct_offset, rtdose_template=rtdose_template)
                    filepath = os.path.join(base_path, p, model, 'rtdose', 'series_2.dcm')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    if convert_moved or not os.path.exists(filepath):
                        rtdose_dicom.save_as(filepath)

def create_registered_dataset(
    dataset: str,
    dest_dataset: str,
    fill: Union[float, Literal['min']] = -2000,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[LandmarkIDs] = 'all',
    moving_study_id: str = 'study_0',
    regions: Optional[Regions] = 'all',
    **kwargs) -> None:
    logging.arg_log('Creating registered dataset', ('dataset', 'dest_dataset'), (dataset, dest_dataset))

    # Create dest dataset.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })
    dest_set = recreate_nifti(dest_dataset)
    write_flag(dest_set, f'__REGISTERED_FROM_{dataset}')

    # Copy dataset files.
    filepath = os.path.join(set.path, 'splits.csv')
    if os.path.exists(filepath):
        destpath = os.path.join(dest_set.path, 'splits.csv')
        shutil.copy(filepath, destpath)

    # Copy patient data.
    pat_ids = set.list_patients()
    for p in tqdm(pat_ids):
        # Perform rigid registration.
        moved_ct, moved_region_data, moved_landmark_data, transform = rigid_registration(dataset, p, moving_study_id, p, fixed_study_id, fill=fill, landmarks=landmarks, regions=regions, regions_ignore_missing=True, **kwargs)

        # Don't do this - it messes up visualisation. Just do it during training/inference.
        # # Fill in padding values.
        # # The fixed CT may be padded, and we should replace values in the moved CT with these padded values.
        # # We couldn't do this earlier, as the moving CT may not have been aligned with the fixed CT.
        pat = set.patient(p)
        fixed_study = pat.study('study_1')
        fixed_ct = fixed_study.ct_data 
        # moved_ct[fixed_ct == pad_value] = pad_value

        # Save moved data.
        fixed_spacing = fixed_study.ct_spacing
        fixed_offset = fixed_study.ct_offset
        moved_study = dest_set.patient(p, check_path=False).study(moving_study_id, check_path=False)
        filepath = os.path.join(moved_study.path, 'ct', 'series_0.nii.gz')
        save_nifti(moved_ct, filepath, spacing=fixed_spacing, offset=fixed_offset)

        if moved_region_data is not None:
            for r, moved_r in moved_region_data.items():
                filepath = os.path.join(moved_study.path, 'regions', 'series_1', f'{r}.nii.gz')
                save_nifti(moved_r, filepath, spacing=fixed_spacing, offset=fixed_offset)

        landmark_cols = ['landmark-id', 0, 1, 2]    # Don't save patient-id/study-id cols.
        if moved_landmark_data is not None:
            filepath = os.path.join(moved_study.path, 'landmarks', 'series_1.csv')
            save_csv(moved_landmark_data[landmark_cols], filepath)

        # Save transform - we'll need this to propagate fixed landmarks to non-registered moving images.
        # TRE is typically calculated in the moving image space. 
        dest_fixed_study = dest_set.patient(p, check_path=False).study(fixed_study_id, check_path=False)
        filepath = os.path.join(dest_fixed_study.path, 'dvf', 'series_0.tfm')
        sitk_save_transform(transform, filepath)

        # Add fixed data.
        # # Moved CT will have introduced 'padding' values, that should not be matched to "real" intensities.
        # # We need to add these padding values to the fixed CT, so that the network is not confused.
        # if pad_value is not None:
        #     fixed_ct[moved_ct == pad_value] = pad_value
        filepath = os.path.join(dest_fixed_study.path, 'ct', 'series_1.nii.gz')
        save_nifti(fixed_ct, filepath, spacing=fixed_spacing, offset=fixed_offset)

        if regions is not None:
            fixed_region_data = fixed_study.region_data(regions=regions, regions_ignore_missing=True)
            if fixed_region_data is not None:
                for r, fixed_r in fixed_region_data.items():
                    filepath = os.path.join(dest_fixed_study.path, 'regions', 'series_1', f'{r}.nii.gz')
                    save_nifti(fixed_r, filepath, spacing=fixed_spacing, offset=fixed_offset)

        if landmarks is not None:
            fixed_landmark_data = fixed_study.landmark_data(landmarks=landmarks)
            if fixed_landmark_data is not None:
                filepath = os.path.join(dest_fixed_study.path, 'landmarks', 'series_1.csv')
                save_csv(fixed_landmark_data[landmark_cols], filepath)
