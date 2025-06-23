import os
from tqdm import tqdm
from typing import *

from mymi.datasets import DicomDataset, NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import resample, save_sitk_transform, sitk_transform_points, velocity_load_transform
from mymi.typing import *
from mymi.utils import *

def convert_velocity_predictions_to_nifti(
    dataset: str,
    fixed_study_id: str = 'study_1',
    moving_study_id: str = 'study_0',
    landmarks: Optional[Landmarks] = 'all',
    pat_prefix: Optional[str] = None,
    regions: Optional[Regions] = 'all',
    transform_types: List[str] = ['dmp', 'edmp']) -> None:
    dicom_set = DicomDataset(dataset)
    nifti_set = NiftiDataset(dataset)
    pat_ids = dicom_set.list_patients()
    for p in tqdm(pat_ids):
        # Remove prefix.
        if pat_prefix is not None:
            p_dest = p.replace(pat_prefix, '')
        else:
            p_dest = p

        # Load fixed/moving images.
        moving_study = nifti_set.patient(p_dest).study(moving_study_id)
        moving_ct = moving_study.ct_data
        moving_spacing = moving_study.ct_spacing
        moving_offset = moving_study.ct_offset
        fixed_study = nifti_set.patient(p_dest).study(fixed_study_id)
        fixed_spacing = fixed_study.ct_spacing
        fixed_offset = fixed_study.ct_offset

        for t in transform_types:
            # Load velocity '.bdf' as sitk transform
            model = f'velocity-{t}'
            dvf_path = os.path.join(dicom_set.path, 'data', 'velocity', p, f'{model}.bdf')
            if not os.path.exists(dvf_path):
                logging.info(f"Skipping prediction '{dvf_path}' - not found.")
                continue
            # Note that the 'affine' transform pre-alignment is built into the velocity DVF transform.
            transform = velocity_load_transform(dvf_path, fixed_offset)

            # Move CT image.
            moved_ct = resample(moving_ct, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                
            # Save moved CT.
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, fixed_study_id, p_dest, moving_study_id, 'ct', f'{model}.nii.gz')
            save_nifti(moved_ct, filepath, spacing=fixed_spacing, offset=fixed_offset)

            # Save warp.
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, fixed_study_id, p_dest, moving_study_id, 'dvf', f'{model}.hdf5')
            save_sitk_transform(transform, filepath)

            # Move regions.
            if regions is not None:
                regions = regions_to_list(regions, literals={ 'all': moving_study.list_regions })
                for r in regions:
                    if not moving_study.has_regions(r):
                        continue
                    moving_region = moving_study.region_data(regions=r)[r]
                    moved_region = resample(moving_region, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                    filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, fixed_study_id, p_dest, moving_study_id, 'regions', r, f'{model}.nii.gz')
                    save_nifti(moved_region, filepath, spacing=fixed_spacing, offset=fixed_offset)

            # Move landmarks.
            # We move 'fixed' to 'moving' for TRE calculation to avoid finding inverse of potentially
            # non-invertible transform
            lm_df = fixed_study.landmark_data(landamrks=landmarks)
            lm_data = lm_df[list(range(3))].to_numpy()
            lm_data_t = sitk_transform_points(lm_data, transform) 
            lm_df[list(range(3))] = lm_data_t
            # These are redundant columns, and shouldn't be stored on disk. They should be added at load time.
            lm_df = lm_df.drop(columns=['patient-id', 'study-id'])
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, fixed_study_id, p_dest, moving_study_id, 'landmarks', f'{model}.csv')
            save_files_csv(lm_df, filepath, overwrite=True)
