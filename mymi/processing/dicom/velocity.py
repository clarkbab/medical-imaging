import os
from tqdm import tqdm
from typing import *

from mymi.datasets import DicomDataset, NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import resample, sitk_save_transform, sitk_transform_points, load_velocity_transform
from mymi.typing import *
from mymi.utils import *

def convert_velocity_registrations_to_nifti(
    dataset: str,
    fixed_study: StudyID = 'study_1',
    landmark: Optional[LandmarkIDs] = 'all',
    landmarks_series: SeriesID = 'idx:0',
    method: Union[str, List[str]] = ['dmp', 'edmp', 'rir', 'sg_c', 'sg_lm'],
    moving_study: StudyID = 'study_0',
    pat: PatientIDs = 'all',
    pat_prefix: Optional[str] = None,
    region: Optional[RegionIDs] = 'all',
    ) -> None:
    methods = arg_to_list(method, str)
    dicom_set = DicomDataset(dataset)
    nifti_set = NiftiDataset(dataset)
    pat_ids = dicom_set.list_patients(pat=pat)
    fixed_study_id = fixed_study
    moving_study_id = moving_study
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
        moving_origin = moving_study.ct_origin
        fixed_study = nifti_set.patient(p_dest).study(fixed_study_id)
        fixed_size = fixed_study.ct_data.shape
        fixed_spacing = fixed_study.ct_spacing
        fixed_origin = fixed_study.ct_origin

        for m in methods:
            # Load velocity 'REG' exported file.
            reg_path = os.path.join(dicom_set.path, 'data', 'velocity', p, f'{m}.dcm')
            if not os.path.exists(reg_path):
                logging.info(f"Skipping prediction '{reg_path}' - not found.")
                continue
            transform = load_velocity_transform(reg_path)

            # Move CT image.
            moved_ct = resample(moving_ct, origin=moving_origin, output_origin=fixed_origin, output_size=fixed_size, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                
            # Save moved CT.
            model = f"velocity-{m}"
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', 'patients', p_dest, fixed_study.id, p_dest, moving_study.id, 'ct', f'{model}.nii.gz')
            save_nifti(moved_ct, filepath, spacing=fixed_spacing, origin=fixed_origin)

            # Save warp.
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', 'patients', p_dest, fixed_study.id, p_dest, moving_study.id, 'transform', f'{model}.hdf5')
            sitk_save_transform(transform, filepath)

            # Move regions.
            if region is not None:
                regions = regions_to_list(region, literals={ 'all': moving_study.list_regions })
                for r in regions:
                    if not moving_study.has_region(r):
                        continue
                    moving_region = moving_study.regions_data(region=r)[r]
                    moved_region = resample(moving_region, origin=moving_origin, output_origin=fixed_origin, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                    filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', 'patients', p_dest, fixed_study.id, p_dest, moving_study.id, 'regions', r, f'{model}.nii.gz')
                    save_nifti(moved_region, filepath, spacing=fixed_spacing, origin=fixed_origin)

            # Move landmarks.
            # We move 'fixed' to 'moving' for TRE calculation to avoid finding inverse of potentially
            # non-invertible transform
            fixed_lm_series = fixed_study.landmarks_series(landmarks_series)
            print(fixed_lm_series.filepath)
            lm_df = fixed_lm_series.data(landmark=landmark)
            lm_data = lm_df[list(range(3))].to_numpy()
            lm_data_t = sitk_transform_points(lm_data, transform) 
            lm_df[list(range(3))] = lm_data_t
            # These are redundant columns, and shouldn't be stored on disk. They should be added at load time.
            lm_df = lm_df.drop(columns=['patient-id', 'study-id'])
            filepath = os.path.join(nifti_set.path, 'data', 'predictions', 'registration', 'patients', p_dest, fixed_study.id, p_dest, moving_study.id, 'landmarks', f'{model}.csv')
            save_csv(lm_df, filepath, overwrite=True)
