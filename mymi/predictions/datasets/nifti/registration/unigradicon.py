import numpy as np
import os
import subprocess
from tqdm import tqdm
from typing import Optional

from mymi.datasets.nifti import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import sitk_load_transform, sitk_transform_points
from mymi.typing import PatientLandmarks, PatientRegions
from mymi.utils import save_csv

def create_unigradicon_predictions(
    dataset: str,
    model: str,
    register_ct: bool = True,
    landmarks: Optional[PatientLandmarks] = 'all',
    regions: Optional[PatientRegions] = 'all',
    use_io: bool = False) -> None:
    logging.arg_log('Making UniGradICON predictions', ('dataset', 'model', 'regions'), (dataset, model, regions))

    # Load patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    for p in tqdm(pat_ids):
        # Load details.
        moving_pat_id, moving_study_id = p, 'study_0'
        fixed_pat_id, fixed_study_id = p, 'study_1'
        # moving/fixed_series_id = 'series_0' implicit.
        moving_pat = set.patient(moving_pat_id)
        fixed_pat = set.patient(fixed_pat_id)
        moving_study = moving_pat.study(moving_study_id)
        fixed_study = fixed_pat.study(fixed_study_id)
        reg_path = os.path.join(set.path, 'data', 'predictions', 'registration', moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id)
        moved_path = os.path.join(reg_path, 'ct', f'{model}.nii.gz')
        transform_path = os.path.join(reg_path, 'dvf', f'{model}.hdf5')

        # Register CT images.
        if register_ct:
            os.makedirs(os.path.dirname(moved_path), exist_ok=True)
            os.makedirs(os.path.dirname(transform_path), exist_ok=True)
            io_iterations = 50 if use_io else None
            command = [
                'unigradicon-register',
                '--moving', moving_study.ct_path,
                '--moving_modality', 'ct',
                '--fixed', fixed_study.ct_path,
                '--fixed_modality', 'ct',
                '--warped_moving_out', moved_path,
                '--transform_out', transform_path,
                '--io_iterations', str(io_iterations)
            ]
            logging.info(command)
            subprocess.run(command)

        # Register regions.
        if regions is not None:
            regions = regions_to_list(regions, literals={ 'all': set.list_regions })
            for r in regions:
                if not moving_study.has_regions(r):
                    continue

                # Perform region warp.
                moving_region_path = moving_study.region_path(r)
                moved_region_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                os.makedirs(os.path.dirname(moved_region_path), exist_ok=True)
                os.makedirs(os.path.dirname(transform_path), exist_ok=True)
                command = [
                    'unigradicon-warp',
                    '--moving', moving_region_path,
                    '--fixed', fixed_study.ct_path,
                    '--transform', transform_path,
                    '--warped_moving_out', moved_region_path,
                    '--nearest_neighbor'
                ]
                logging.info(command)
                subprocess.run(command)

        # Transform any fixed landmarks back to moving space.
        if landmarks is not None:
            transform = sitk_load_transform(transform_path)
            fixed_lm_df = fixed_study.landmark_data(landmarks=landmarks)
            lm_data = fixed_lm_df[list(range(3))].to_numpy()
            lm_data_t = sitk_transform_points(lm_data, transform)
            if np.allclose(lm_data_t, lm_data):
                logging.warning(f"Moved points are very similar to fixed points - identity transform?")
            moving_lm_df = fixed_lm_df.copy()
            moving_lm_df[list(range(3))] = lm_data_t

            # Save transformed points.
            filepath = os.path.join(reg_path, 'landmarks', f'{model}.csv')
            save_csv(moving_lm_df, filepath, overwrite=True)
