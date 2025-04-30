import os
import shutil
from tqdm import tqdm
from typing import *

from mymi.datasets.nifti import NiftiDataset, recreate as recreate_nifti
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import sitk_save_transform
from mymi.transforms.dataset.nifti import rigid_registration
from mymi.typing import Landmarks, Regions
from mymi.utils import save_as_nifti, save_files_csv

from ...processing import write_flag

def create_registered_dataset(
    dataset: str,
    dest_dataset: str,
    fill: Union[float, Literal['min']] = -2000,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
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
    filepath = os.path.join(set.path, 'holdout-split.csv')
    if os.path.exists(filepath):
        destpath = os.path.join(dest_set.path, 'holdout-split.csv')
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
        save_as_nifti(moved_ct, fixed_spacing, fixed_offset, filepath)

        if moved_region_data is not None:
            for r, moved_r in moved_region_data.items():
                filepath = os.path.join(moved_study.path, 'regions', 'series_1', f'{r}.nii.gz')
                save_as_nifti(moved_r, fixed_spacing, fixed_offset, filepath)

        landmark_cols = ['landmark-id', 0, 1, 2]    # Don't save patient-id/study-id cols.
        if moved_landmark_data is not None:
            filepath = os.path.join(moved_study.path, 'landmarks', 'series_1.csv')
            save_files_csv(moved_landmark_data[landmark_cols], filepath)

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
        save_as_nifti(fixed_ct, fixed_spacing, fixed_offset, filepath)

        if regions is not None:
            fixed_region_data = fixed_study.region_data(regions=regions, regions_ignore_missing=True)
            if fixed_region_data is not None:
                for r, fixed_r in fixed_region_data.items():
                    filepath = os.path.join(dest_fixed_study.path, 'regions', 'series_1', f'{r}.nii.gz')
                    save_as_nifti(fixed_r, fixed_spacing, fixed_offset, filepath)

        if landmarks is not None:
            fixed_landmark_data = fixed_study.landmark_data(landmarks=landmarks)
            if fixed_landmark_data is not None:
                filepath = os.path.join(dest_fixed_study.path, 'landmarks', 'series_1.csv')
                save_files_csv(fixed_landmark_data[landmark_cols], filepath)
