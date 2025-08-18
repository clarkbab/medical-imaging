import os
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi.datasets.training import recreate
from mymi.loaders import get_holdout_split
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import pad, resample
from mymi.typing import *
from mymi.utils import *

def convert_to_registration_training_holdout(
    dataset: str,
    dest_dataset: Optional[str] = None,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_study_id: str = 'study_0',
    normalise: bool = False,
    norm_mean: Optional[float] = None,
    norm_stdev: Optional[float] = None,
    regions: Optional[Regions] = 'all',
    size_factor: Optional[int] = None,
    spacing: Optional[Spacing3D] = None,
    **kwargs) -> None:
    logging.arg_log('Converting NIFTI dataset to registration holdout TRAINING', ('dataset', 'spacing', 'regions', 'landmarks'), (dataset, spacing, regions, landmarks))
    start = time()

    # Check params.
    if normalise:
        assert norm_mean is not None
        assert norm_stdev is not None

    # Get regions.
    set = NiftiDataset(dataset)
    landmarks = arg_to_list(landmarks, int, literals={ 'all': set.list_landmarks })
    if len(landmarks) == 0:
        landmarks = None
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })
    if len(regions) == 0:
        regions = None

    # Create training dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    dest_set = recreate(dest_dataset)

    # Write params.
    cols = {
        'param': str,
        'value': str
    }
    df = pd.DataFrame(columns=cols.keys())

    label_types = []    # Can get fixed image from the inputs.
    if regions is not None:
        label_types += ['regions'] * 2   # Fixed and moving.
    if landmarks is not None:
        label_types += ['landmarks'] * 2   # Fixed and moving.
    params = [
        { 'param': 'type', 'value': 'holdout' },
        { 'param': 'spacing', 'value': str(spacing) },
        { 'param': 'regions', 'value': str(regions) },
        { 'param': 'landmarks', 'value': str(landmarks) },
        { 'param': 'label-types', 'value': label_types },
    ]
    for p in params:
        df = append_row(df, p)
    filepath = os.path.join(dest_set.path, 'params.csv')
    df.to_csv(filepath, index=False)

    cols = {
        'split': str,
        'sample-id': int,
        'origin-dataset': str,
        'origin-patient-id': str,
        'origin-fixed-study-id': str,
        'origin-moving-study-id': str,
    }
    df = pd.DataFrame(columns=cols.keys())

    # Write each split to the dataset.
    pat_ids = get_holdout_split(dataset, **kwargs)
    splits = ['train', 'validate', 'test']
    sample_id_counter = 0
    for s, ps in tqdm(zip(splits, pat_ids)):
        for p in tqdm(ps, leave=False):
            sample_id = f"{sample_id_counter:03}"

            # Load fixed data.
            pat = set.patient(p)
            fixed_study = pat.study(fixed_study_id)
            fixed_ct = fixed_study.ct_data
            fixed_spacing = fixed_study.ct_spacing
            if landmarks is not None:
                fixed_landmarks = fixed_study.landmark_data(landmarks=landmarks)
            if regions is not None:
                fixed_region_data = fixed_study.region_data(regions=regions)
            
            # Load moving data.
            moving_study = pat.study(moving_study_id)
            moving_ct = moving_study.ct_data
            assert moving_study.ct_spacing == fixed_spacing   # Images are pre-registered.
            if landmarks is not None:
                moving_landmarks = moving_study.landmark_data(landmarks=landmarks)
            if regions is not None:
                moving_region_data = moving_study.region_data(regions=regions)

            # # Normalise CT data. - Let the dataloader do this so we don't have to manually
            # calculate for each training dataset. For example, values will differ depending on 
            # training dataset resolution (444, 222, 112, for example).
            # if normalise:
            #     fixed_ct = (fixed_ct - norm_mean) / norm_stdev
            #     moving_ct = (moving_ct - norm_mean) / norm_stdev

            if spacing:
                # Resample fixed data.
                fixed_ct = resample(fixed_ct, spacing=fixed_spacing, output_spacing=spacing)
                if regions is not None:
                    for r, d in fixed_region_data.items():
                        fixed_region_data[r] = resample(d, spacing=fixed_spacing, output_spacing=spacing)
                # Landmarks shouldn't be resampled! Resampling just changes discrete measurements in physical space,
                # landmarks are already in physical space.
                # if landmarks is not None:
                #     fixed_landmarks = resample_landmarks(fixed_landmarks, spacing=fixed_spacing, output_spacing=spacing)

                # Resample moving data.
                moving_ct = resample(moving_ct, spacing=fixed_spacing, output_spacing=spacing)
                if regions is not None:
                    for r, d in moving_region_data.items():
                        moving_region_data[r] = resample(d, spacing=fixed_spacing, output_spacing=spacing)
                # if landmarks is not None:
                #     moving_landmarks = resample_landmarks(moving_landmarks, spacing=fixed_spacing, output_spacing=spacing)

            if size_factor is not None:
                # Pad fixed data to ensure divisible by 2 ** size_factor.
                fixed_ct_size = fixed_ct.shape
                factors = np.array(fixed_ct_size) / (2 ** size_factor)
                new_fixed_ct_size = tuple((np.ceil(factors) * 2 ** size_factor).astype(int))
                if new_fixed_ct_size != fixed_ct_size:
                    pad_box = ((0, 0, 0), new_fixed_ct_size)
                    fixed_ct = pad(fixed_ct, pad_box)
                    for r, d in fixed_region_data.items():
                        fixed_region_data[r] = pad(d, pad_box)

                # Pad moving data to ensure divisible by 2 ** size_factor.
                moving_ct_size = moving_ct.shape
                factors = np.array(moving_ct_size) / (2 ** size_factor)
                new_moving_ct_size = tuple((np.ceil(factors) * 2 ** size_factor).astype(int))
                if new_moving_ct_size != moving_ct_size:
                    pad_box = ((0, 0, 0), new_moving_ct_size)
                    moving_ct = pad(moving_ct, pad_box)
                    for r, d in moving_region_data.items():
                        moving_region_data[r] = pad(d, pad_box)

                # TODO: Pad landmarks.

            # Don't do this - we're eliminating crucial information from the fixed image (or moving), for
            # example fixed images (inhale) typically have the diaphragm lower than the moving image, and 
            # the diaphragm will be cropped from the fixed image and can't be mapped to moving.
            # # Propagate "padding" between fixed and moving images.
            # # Could we store "padding" as a mask?
            # # If we cropped images, instead of using a padding value, this needs to be before registration, 
            # # which will rotate images and introduce "padding" values - that also have no physical meaning.
            # # We can't crop them as we'd lose valuable image information from other voxels on the same slice.
            # # So at some point we need to deal with "padding" values.
            # # Any stored mask would just be used to convert image values to "padding" values before training,
            # # and would have increased overhead of managing separate masks in addition to images.
            # if pad_value is not None:
            #     fixed_ct[moving_ct == pad_value] = pad_value
            #     moving_ct[fixed_ct == pad_value] = pad_value

            # Save input (fixed/moving images).
            input = np.stack((fixed_ct, moving_ct))
            filepath = os.path.join(dest_set.path, 'data', s, 'inputs', f"{sample_id}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=input)

            # # Save fixed image (label 0).
            # filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}-0.npz")
            # os.makedirs(os.path.dirname(filepath), exist_ok=True)
            # np.savez_compressed(filepath, data=fixed_ct)
            
            output_n = 0
            if regions is not None:
                # Load fixed region data.
                n_channels = len(regions) + 1
                label = np.zeros((n_channels, *fixed_ct.shape), dtype=bool)
                mask = np.zeros((n_channels), dtype=bool)
                for r, d in fixed_region_data.items():
                    channel = regions.index(r) + 1
                    label[channel] = d
                    mask[channel] = True

                # Add background if all foreground classes present. 
                if pat.has_region(regions, all=True):
                    label[0] = np.invert(label.any(axis=0))
                    mask[0] = True

                # Save fixed region data.
                filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}-{output_n}.npz")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.savez_compressed(filepath, data=label)
                filepath = os.path.join(dest_set.path, 'data', s, 'masks', f"{sample_id}-{output_n}.npz")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.savez_compressed(filepath, data=mask, exist_ok=True)
                output_n += 1
                
                # Load moving region data.
                n_channels = len(regions) + 1
                label = np.zeros((n_channels, *fixed_ct.shape), dtype=bool)
                mask = np.zeros((n_channels), dtype=bool)
                for r, d in fixed_region_data.items():
                    channel = regions.index(r) + 1
                    label[channel] = d
                    mask[channel] = True

                # Add background if all foreground classes present. 
                if pat.has_region(regions, all=True):
                    label[0] = np.invert(label.any(axis=0))
                    mask[0] = True

                # Save moving region data.
                filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}-{output_n}.npz")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.savez_compressed(filepath, data=label)
                filepath = os.path.join(dest_set.path, 'data', s, 'masks', f"{sample_id}-{output_n}.npz")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.savez_compressed(filepath, data=mask, exist_ok=True)
                output_n += 1

            if landmarks is not None:
                # Save fixed landmark data.
                landmark_cols = ['landmark-id', 0, 1, 2]    # Don't save patient-id/study-id.
                filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}-{output_n}.csv")
                save_csv(fixed_landmarks[landmark_cols], filepath)
                output_n += 1

                # Save moving landmark data.
                filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}-{output_n}.csv")
                save_csv(moving_landmarks[landmark_cols], filepath)
                output_n += 1

            # Add index entry.
            data = {
                'split': s,
                'sample-id': sample_id,
                'origin-dataset': dataset,
                'origin-patient-id': p,
                'origin-fixed-study-id': fixed_study.id,
                'origin-moving-study-id': moving_study.id,
            }
            df = append_row(df, data)

            sample_id_counter += 1

    # Write index.
    df = df.astype(cols)
    filepath = os.path.join(dest_set.path, 'index.csv')
    df.to_csv(filepath, index=False)

    # Record processing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    path = os.path.join(dest_set.path, f'__CONVERT_FROM_NIFTI_END_MINS_{mins}__')
    Path(path).touch()

def convert_to_segmentation_training_holdout(
    dataset: str,
    dest_dataset: Optional[str] = None,
    normalise: bool = False,
    norm_mean: Optional[float] = None,
    norm_stdev: Optional[float] = None,
    regions: Optional[Regions] = 'all',
    size_factor: Optional[int] = None,
    spacing: Optional[Spacing3D] = None,
    **kwargs) -> None:
    start = time()
    logging.arg_log('Converting NIFTI dataset to holdout TRAINING', ('dataset', 'spacing', 'regions'), (dataset, spacing, regions))
    if normalise:
        raise ValueError('Normalisation how handled by data loader.')

    # Check params.
    if normalise:
        assert norm_mean is not None
        assert norm_stdev is not None

    # Get regions.
    set = NiftiDataset(dataset)
    regions = regions_to_list(regions, literals={ 'all': set.list_regions })

    # Create training dataset.
    dest_dataset = dataset if dest_dataset is None else dest_dataset
    dest_set = recreate(dest_dataset)

    # Write params.
    cols = {
        'param': str,
        'value': str
    }
    df = pd.DataFrame(columns=cols.keys())

    label_types = ['regions']
    params = [
        { 'param': 'type', 'value': 'holdout' },
        { 'param': 'spacing', 'value': str(spacing) },
        { 'param': 'regions', 'value': str(regions) },
        { 'param': 'label-types', 'value': str(label_types) },
    ]
    for p in params:
        df = append_row(df, p)
    filepath = os.path.join(dest_set.path, 'params.csv')
    df.to_csv(filepath, index=False)

    # Load patients.
    pat_ids = set.list_patients(regions=regions)

    cols = {
        'split': str,
        'sample-id': int,
        'origin-dataset': str,
        'origin-patient-id': str,
        'origin-study-id': str,
    }
    df = pd.DataFrame(columns=cols.keys())

    # Write each split to the dataset.
    pat_ids = get_holdout_split(dataset, **kwargs)
    splits = ['train', 'validate', 'test']
    sample_id_counter = 0
    for s, ps in tqdm(zip(splits, pat_ids)):
        for p in tqdm(ps, leave=False):
            sample_id = f"{sample_id_counter:03}"
            
            # Load sample data.
            pat = set.patient(p)
            study = pat.default_study
            ct_data = study.ct_data
            ct_spacing = study.ct_spacing
            region_data = study.region_data(regions=regions)

            # Normalise CT data.
            if normalise:
                ct_data = (ct_data - norm_mean) / norm_stdev

            # Resample data.
            if spacing:
                ct_data = resample(ct_data, spacing=ct_spacing, output_spacing=spacing)
                for r, d in region_data.items():
                    region_data[r] = resample(d, spacing=ct_spacing, output_spacing=spacing)

            # Pad data to ensure divisible by 2 ** size_factor.
            if size_factor is not None:
                ct_size = ct_data.shape
                factors = np.array(ct_size) / (2 ** size_factor)
                new_ct_size = tuple((np.ceil(factors) * 2 ** size_factor).astype(int))
                if new_ct_size != ct_size:
                    pad_box = ((0, 0, 0), new_ct_size)
                    ct_data = pad(ct_data, pad_box)
                    for r, d in region_data.items():
                        region_data[r] = pad(d, pad_box)

            # Save input data.
            filepath = os.path.join(dest_set.path, 'data', s, 'inputs', f"{sample_id}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=ct_data)
            
            # Create/save output data.
            n_channels = len(regions) + 1
            label = np.zeros((n_channels, *ct_data.shape), dtype=bool)
            mask = np.zeros((n_channels), dtype=bool)
            for r, d in region_data.items():
                channel = regions.index(r) + 1
                label[channel] = d
                mask[channel] = True

            # Add background if all foreground classes present. 
            if pat.has_region(regions):
                label[0] = np.invert(label.any(axis=0))
                mask[0] = True

            # Save label data.
            filepath = os.path.join(dest_set.path, 'data', s, 'labels', f"{sample_id}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=label)
            filepath = os.path.join(dest_set.path, 'data', s, 'masks', f"{sample_id}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez_compressed(filepath, data=mask, exist_ok=True)

            # Add index entry.
            data = {
                'split': s,
                'sample-id': sample_id,
                'origin-dataset': dataset,
                'origin-patient-id': p,
                'origin-study-id': study.id,
            }
            df = append_row(df, data)

            sample_id_counter += 1

    # Write index.
    df = df.astype(cols)
    filepath = os.path.join(dest_set.path, 'index.csv')
    df.to_csv(filepath, index=False)

    # Record processing time.
    end = time()
    mins = int(np.ceil((end - start) / 60))
    path = os.path.join(dest_set.path, f'__CONVERT_FROM_NIFTI_END_MINS_{mins}__')
    Path(path).touch()
