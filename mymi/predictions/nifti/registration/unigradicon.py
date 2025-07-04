import numpy as np
import os
import SimpleITK as sitk
import subprocess
from tqdm import tqdm
from typing import *

from mymi.datasets.nifti import NiftiDataset
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import load_itk_transform, load_sitk_transform, resample, save_sitk_transform, sitk_transform_points
from mymi.typing import *
from mymi.utils import *

def create_unigradicon_predictions(
    dataset: str,
    model: str,
    register_ct: bool = True,
    landmarks: Optional[Landmarks] = 'all',
    regions: Optional[Regions] = 'all',
    splits: Optional[Split] = 'all',
    use_io: bool = False,
    use_timing: bool = True) -> None:
    logging.arg_log('Making UniGradICON predictions', ('dataset', 'model', 'regions'), (dataset, model, regions))

    # Create timing table.
    if use_timing:
        cols = {
            'dataset': str,
            'patient-id': str,
            'device': str
        }
        timer = Timer(cols)

    # Load patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(splits=splits)

    for p in tqdm(pat_ids):
        # Timing table data.
        data = {
            'dataset': dataset,
            'patient-id': p,
            'device': 'cuda',
        }
        with timer.record(data, enabled=use_timing):
            # Load details.
            moving_pat_id, moving_study_id = p, 'study_0'
            fixed_pat_id, fixed_study_id = p, 'study_1'
            # moving/fixed_series_id = 'series_0' implicit.
            moving_pat = set.patient(moving_pat_id)
            fixed_pat = set.patient(fixed_pat_id)
            moving_study = moving_pat.study(moving_study_id)
            fixed_study = fixed_pat.study(fixed_study_id)
            reg_path = os.path.join(set.path, 'data', 'predictions', 'registration', fixed_pat_id, fixed_study_id, moving_pat_id, moving_study_id)
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

                # Convert transform to sitk.
                t_itk = load_itk_transform(transform_path)[0]
                transform = convert_transform_to_sitk(t_itk)
                save_sitk_transform(transform, transform_path)

                # Apply transform to moving CT.
                fixed_ct = fixed_study.ct_data
                fixed_spacing = fixed_study.ct_spacing
                fixed_offset = fixed_study.ct_offset
                moving_ct = moving_study.ct_data
                moving_spacing = moving_study.ct_spacing
                moving_offset = moving_study.ct_offset
                moved_ct = resample(moving_ct, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                save_nifti(moved_ct, moved_path, spacing=fixed_spacing, offset=fixed_offset)

            # Register regions.
            if regions is not None:
                regions = regions_to_list(regions, literals={ 'all': set.list_regions })
                for r in regions:
                    if not moving_study.has_regions(r):
                        continue

                    # Load data.
                    fixed_ct = fixed_study.ct_data
                    fixed_spacing = fixed_study.ct_spacing
                    fixed_offset = fixed_study.ct_offset
                    moving_label = moving_study.regions_data(r)[r]
                    moving_spacing = moving_study.ct_spacing
                    moving_offset = moving_study.ct_offset
                    transform = load_sitk_transform(transform_path)

                    # Perform transform.
                    moved_label = resample(moving_label, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
                    moved_label_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                    os.makedirs(os.path.dirname(moved_label_path), exist_ok=True)
                    save_nifti(moved_label, moved_label_path, spacing=fixed_spacing, offset=fixed_offset)

                    # # Perform region warp.
                    # moving_region_path = moving_study.region_path(r)
                    # moved_region_path = os.path.join(reg_path, 'regions', r, f'{model}.nii.gz')
                    # os.makedirs(os.path.dirname(moved_region_path), exist_ok=True)
                    # command = [
                    #     'unigradicon-warp',
                    #     '--moving', moving_region_path,
                    #     '--fixed', fixed_study.ct_path,
                    #     '--transform', transform_path,
                    #     '--warped_moving_out', moved_region_path,
                    #     '--nearest_neighbor'
                    # ]
                    # logging.info(command)
                    # subprocess.run(command)

            # Transform any fixed landmarks back to moving space.
            if landmarks is not None:
                transform = load_sitk_transform(transform_path)
                if fixed_study.has_landmarks(landmarks=landmarks):
                    fixed_lm_df = fixed_study.landmarks_data(landmarks=landmarks)
                    lm_data = fixed_lm_df[list(range(3))].to_numpy()
                    lm_data_t = sitk_transform_points(lm_data, transform)
                    if np.allclose(lm_data_t, lm_data):
                        logging.warning(f"Moved points are very similar to fixed points - identity transform?")
                    moving_lm_df = fixed_lm_df.copy()
                    moving_lm_df[list(range(3))] = lm_data_t

                    # Save transformed points.
                    filepath = os.path.join(reg_path, 'landmarks', f'{model}.csv')
                    save_csv(moving_lm_df, filepath, overwrite=True)

    # Save timing data.
    if use_timing:
        filepath = os.path.join(set.path, 'data', 'predictions', 'registration', 'timing', f'{model}.csv')
        timer.save(filepath)

# Because ITK/SimpleITK load nifti files with negative x/y directions and offsets,
# the transform is configured to work with this input space. Which is different from
# how nibabel loads nifti files. ITK expects nifti files to use RAS but we write them
# using LPS.
# Here we flip the transform components to work with nibabel/our coordinate system.
def convert_transform_to_sitk(t: itk.Transform) -> sitk.Transform:
    # Applied in reverse order in composite transform.
    t0 = itk.down_cast(t.GetNthTransform(0))    # Affine 2 - network to image space
    t1 = itk.down_cast(t.GetNthTransform(1))    # DVF - network to network space
    t2 = itk.down_cast(t.GetNthTransform(2))    # Affine 1 - image to network space

    affine_transforms = [t0, t2]

    new_ts = []
    for at in affine_transforms:
        new_at = itk_centred_affine_to_sitk(at)
        new_ts.append(new_at)

    # Convert DVF to sitk.
    dvf_image_itk = t1.GetDisplacementField()
    dvf_data, dvf_spacing, dvf_offset = from_itk_image(dvf_image_itk)
    # We need to reverse DVF x/y components because our affines are mapping into the negative x/y space.
    dvf_data[0], dvf_data[1] = -dvf_data[0], -dvf_data[1]
    dvf_image_sitk = to_sitk_image(dvf_data, dvf_spacing, dvf_offset, vector=True)
    dir = np.array(dvf_image_sitk.GetDirection())
    dir = reverse_xy(dir)
    dvf_image_sitk.SetDirection(dir)
    dvf_sitk = sitk.DisplacementFieldTransform(dvf_image_sitk)
    new_ts.insert(1, dvf_sitk)

    new_t = sitk.CompositeTransform(new_ts)
    return new_t

def itk_centred_affine_to_sitk(t: itk.CenteredAffineTransform) -> sitk.AffineTransform:
    # Get parameters
    dim = t.GetInputSpaceDimension()
    matrix = np.array(t.GetMatrix()).reshape((dim, dim))
    translation = np.array(t.GetTranslation())
    centre = np.array(t.GetCenter())
    
    # Reverse all x/y params except the scaling.
    # This will map into a network space with negative x/y directions.
    translation[0], translation[1] = -translation[0], -translation[1]
    centre[0], centre[1] = -centre[0], -centre[1]

    # Create a SimpleITK AffineTransform
    sitk_transform = sitk.AffineTransform(dim)

    # Set the center first
    sitk_transform.SetCenter(centre.tolist())

    # Set matrix (flattened)
    sitk_transform.SetMatrix(matrix.flatten().tolist())

    # Set translation
    sitk_transform.SetTranslation(translation.tolist())

    return sitk_transform
