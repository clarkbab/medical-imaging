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
    landmarks: Optional[Landmarks] = None,
    pat_prefix: Optional[str] = None,
    regions: Optional[Regions] = None,
    transform_types: List[str] = ['DMP', 'EDMP']) -> None:
    dicom_set = DicomDataset(dataset)
    nifti_set = NiftiDataset(dataset)
    pat_ids = dicom_set.list_patients()
    pat_ids = [pat_ids[-1]]
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
        fixed_ct = fixed_study.ct_data
        fixed_spacing = fixed_study.ct_spacing
        fixed_offset = fixed_study.ct_offset

        for t in transform_types:
            # Load velocity '.bdf' as sitk transform
            model = f'VELOCITY-{t}'
            transform_path = os.path.join(dicom_set.path, 'data', 'predictions', 'registration', p, moving_study_id, p, fixed_study_id, model, 'dvf.bdf')
            if not os.path.exists(transform_path):
                logging.info(f"Skipping prediction '{transform_path}' - not found.")
                continue
            transform = velocity_load_transform(transform_path, fixed_offset)

            # Move CT image.
            moved_ct = resample(moving_ct, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                
            # Save moved CT.
            modelname = f'VELOCITY-{t}'
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, moving_study_id, p_dest, fixed_study_id, modelname, 'ct', 'series_0.nii.gz')
            save_nifti(moved_ct, filepath, spacing=fixed_spacing, offset=fixed_offset)

            # Save warp.
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, moving_study_id, p_dest, fixed_study_id, modelname, 'dvf', 'series_0.hdf5')
            save_sitk_transform(transform, filepath)

            # Move regions.
            if regions is not None:
                regions = regions_to_list(regions, literals={ 'all', moving_study.list_regions })
                for r in regions:
                    if not moving_study.has_regions(r):
                        continue
                    moving_region = moving_study.region_data(regions=r)[r]
                    moved_region = resample(moving_region, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                    filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, moving_study_id, p_dest, fixed_study_id, modelname, 'regions', 'series_1', f'{r}.nii.gz')
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
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', p_dest, moving_study_id, p_dest, fixed_study_id, modelname, 'landmarks', 'series_1.csv')
            save_files_csv(lm_df, filepath, overwrite=True)
