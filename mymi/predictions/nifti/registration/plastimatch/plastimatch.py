import numpy as np
import os
from tqdm import tqdm

from dicomset import NiftiDataset
from dicomset.utils import create_affine
from mymi import logging
from mymi.predictions.registration import register_plastimatch
from mymi.regions import regions_to_list
from mymi.transforms import resample, sitk_save_transform, sitk_transform_points
from mymi.typing import *
from mymi.utils.io import save_csv
from mymi.utils.nifti import save_nifti
from mymi.utils.timer import Timer

def create_plastimatch_predictions(
    dataset: str,
    container_path: str = '/config/binaries/singularity/containers/plastimatch/1.9.3/plastimatch.sif',
    create_moved_ct: bool = True,
    create_moved_dose: bool = True,
    fixed_study: StudyID = 'study_1',
    landmarks: Optional[LandmarkIDs] = 'all',
    model: str = 'plastimatch',
    moving_study: StudyID = 'study_0',
    pat_ids: PatientIDs = 'all',
    regions: Optional[RegionIDs] = 'all',
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
        print(p)
        # Timing table data.
        data = {
            'dataset': dataset,
            'patient-id': p,
            'device': 'cuda',
        }
        with timer.record(data, enabled=use_timing):
            pat = set.patient(p)
            fixed_study = pat.study(fixed_study)
            moving_study = pat.study(moving_study)

            # Prepare data.
            fixed_ct = fixed_study.ct_data
            moving_ct = moving_study.ct_data
            fixed_affine = create_affine(fixed_study.ct_spacing, fixed_study.ct_origin)
            moving_affine = create_affine(moving_study.ct_spacing, moving_study.ct_origin)

            # Load template.
            template_filepath = os.path.join(os.path.dirname(__file__), 'template.txt')
            with open(template_filepath, 'r') as f:
                template = f.read()

            # Core registration.
            transform = register_plastimatch(
                fixed_ct, moving_ct, fixed_affine, moving_affine,
                container_path=container_path,
                template=template)

            # Save transform.
            pred_base = os.path.join(set.path, 'data', 'predictions', 'registration', 'patients', p, fixed_study.id, p, moving_study.id)
            filepath = os.path.join(pred_base, 'transform', f'{model}.hdf5')
            sitk_save_transform(transform, filepath)

            # Save moved CT.
            if create_moved_ct:
                moved_ct = resample(moving_ct, affine=moving_affine, output_affine=fixed_affine, transform=transform)
                filepath = os.path.join(pred_base, 'ct', f'{model}.nii.gz')
                save_nifti(moved_ct, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

            if regions is not None:
                pat_regions = regions_to_list(regions, literals={ 'all': pat.list_regions })
                for r in pat_regions:
                    moving_label = moving_study.regions_data(regions=r)[r]
                    moved_label = resample(moving_label, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                    filepath = os.path.join(pred_base, 'regions', r, f'{model}.nii.gz')
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    save_nifti(moved_label, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

            if landmarks is not None:
                fixed_lms_df = fixed_study.landmarks_data(landmarks=landmarks)
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

            if create_moved_dose and moving_study.has_dose:
                moving_dose = moving_study.dose_data
                moved_dose = resample(moving_dose, origin=moving_study.ct_origin, output_origin=fixed_study.ct_origin, output_size=fixed_study.ct_size, output_spacing=fixed_study.ct_spacing, spacing=moving_study.ct_spacing, transform=transform)
                filepath = os.path.join(pred_base, 'dose', f'{model}.nii.gz')
                save_nifti(moved_dose, filepath, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
        timer.save(filepath)
