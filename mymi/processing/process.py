import numpy as np
import os
import pandas as pd
import pydicom as dcm
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import Callable, List, Optional, Tuple, Union

from mymi.dataset.training import TrainingDataset, exists as exists_training
from mymi.dataset.training import create as create_training
from mymi.dataset.training import recreate as recreate_training
from mymi.geometry import get_extent
from mymi import logging
from mymi.regions import region_to_list
from mymi.transforms import centre_crop_or_pad_3D, centre_crop_or_pad_4D, crop_3D, crop_4D, resample_3D, resample_4D, top_crop_or_pad_3D
from mymi.types import BoxMM3D, SizeMM3D, Size3D, Spacing3D, ModelName, PatientID, PatientRegion, PatientRegions
from mymi.utils import append_row, arg_to_list, load_csv, save_csv

def convert_brain_crop_to_training(
    set: 'Dataset',
    create_data: bool = True,
    crop_mm: Optional[Union[BoxMM3D, SizeMM3D]] = None,
    dest_dataset: Optional[str] = None,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    load_localiser_prediction: Optional[Callable] = None,
    recreate_dataset: bool = True,
    region: Optional[PatientRegions] = None,
    round_dp: Optional[int] = None,
    spacing: Optional[Spacing3D] = None) -> None:
    logging.arg_log(f'Converting {set.type.name} dataset to TRAINING', ('dataset', 'region'), (set, region))
    regions = region_to_list(region)

    # Use all regions if region is 'None'.
    if regions is None:
        regions = set.list_regions()

    # Create the dataset.
    dest_dataset = set.name if dest_dataset is None else dest_dataset
    if exists_training(dest_dataset):
        if recreate_dataset:
            created = True
            set_t = recreate_training(dest_dataset)
        else:
            created = False
            set_t = TrainingDataset(dest_dataset)
            __destroy_flag(set_t, f'__CONVERT_FROM_{set.type.name}_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(set_t.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        set_t = create_training(dest_dataset)
    __write_flag(set_t, f'__CONVERT_FROM_{set.type.name}_START__')

    # Write params.
    if created:
        filepath = os.path.join(set_t.path, 'params.csv')
        params_df = pd.DataFrame({
            'crop-mm': [str(crop_mm)] if crop_mm is not None else ['None'],
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'regions': [str(regions)],
            'spacing': [str(spacing)] if spacing is not None else ['None'],
        })
        params_df.to_csv(filepath, index=False)
    else:
        for region in regions:
            filepath = os.path.join(set_t.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'crop-mm': [str(crop_mm)] if crop_mm is not None else ['None'],
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'spacing': [str(spacing)] if spacing is not None else ['None'],
                'regions': [str(regions)],
            })
            params_df.to_csv(filepath, index=False)

    # Load patients.
    pat_ids = set.list_patients(region=regions)

    # Get exclusions.
    exc_df = set.excluded_labels

    # Create index.
    cols = {
        'dataset': str,
        'sample-id': int,
        'group-id': float,
        'origin-dataset': str,
        'origin-patient-id': str,
        'region': str,
        'empty': bool
    }
    index = pd.DataFrame(columns=cols.keys())
    index = index.astype(cols)

    # Load patient grouping if present.
    group_df = set.group_index

    # Write each patient to dataset.
    start = time()
    if create_data:
        for i, pat_id in enumerate(tqdm(pat_ids)):
            # Load input data.
            patient = set.patient(pat_id)
            input_spacing = patient.ct_spacing
            input = patient.ct_data

            # Resample input.
            if spacing is not None:
                input = resample_3D(input, spacing=input_spacing, output_spacing=spacing)

            # Crop input.
            if crop_mm is not None:
                # Get crop reference point.
                localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
                brain_label = load_localiser_prediction(set.name, pat_id, localiser)
                if spacing is not None:
                    brain_label = resample_3D(brain_label, spacing=input_spacing, output_spacing=spacing)
                brain_extent = get_extent(brain_label)
                crop_ref = ((brain_extent[0][0] + brain_extent[1][0]) // 2, (brain_extent[0][1] + brain_extent[1][1]) // 2, brain_extent[1][2])

                # Determine if asymmetric (box) or symmetric (size) crop.
                if isinstance(crop_mm[0], tuple):
                    # Perform asymmetric crop.
                    crop_voxels = tuple((tuple(c) for c in (np.array(crop_mm) / spacing).astype(np.int32)))
                    crop = (
                        (int(crop_ref[0] + crop_voxels[0][0]), int(crop_ref[1] + crop_voxels[0][1]), int(crop_ref[2] + crop_voxels[0][2])),
                        (int(crop_ref[0] + crop_voxels[1][0]), int(crop_ref[1] + crop_voxels[1][1]), int(crop_ref[2] + crop_voxels[1][2])),
                    )
                else:
                    # Convert to voxel crop.
                    crop_voxels = tuple((np.array(crop_mm) / np.array(spacing)).astype(np.int32))

                    # Get crop coordinates.
                    # Crop origin is centre-of-extent in x/y, and max-extent in z.
                    # Cropping boundary extends from origin equally in +/- directions for x/y, and extends
                    # in - direction for z.
                    p_above_brain = 0.04
                    crop = (
                        (int(crop_ref[0] - crop_voxels[0] // 2), int(crop_ref[1] - crop_voxels[1] // 2), int(crop_ref[2] - int(crop_voxels[2] * (1 - p_above_brain)))),
                        (int(np.ceil(crop_ref[0] + crop_voxels[0] / 2)), int(np.ceil(crop_ref[1] + crop_voxels[1] / 2)), int(crop_ref[2] + int(crop_voxels[2] * p_above_brain)))
                    )

                # Crop input.
                input = crop_3D(input, crop)

            # Save input.
            __create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat_id).has_region(region):
                    continue

                # Skip if region in 'excluded-labels.csv'.
                if exc_df is not None:
                    pr_df = exc_df[(exc_df['patient-id'] == pat_id) & (exc_df['region'] == region)]
                    if len(pr_df) == 1:
                        continue

                # Load label data.
                label = patient.region_data(region=region)[region]

                # Resample data.
                if spacing is not None:
                    label = resample_3D(label, spacing=input_spacing, output_spacing=spacing)

                # Crop/pad.
                if crop_mm is not None:
                    label = crop_3D(label, crop)

                # Round data after resampling to save on disk space.
                if round_dp is not None:
                    input = np.around(input, decimals=round_dp)

                # Dilate the labels if requested.
                if region in dilate_regions:
                    label = binary_dilation(label, iterations=dilate_iter)

                # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
                if label.sum() != 0:
                    empty = False
                    __create_training_label(set_t, i, label, region=region)
                else:
                    empty = True

                # Add index entry.
                if group_df is not None:
                    tdf = group_df[group_df['patient-id'] == pat_id]
                    if len(tdf) == 0:
                        group_id = np.nan
                    else:
                        assert len(tdf) == 1
                        group_id = tdf.iloc[0]['group-id']
                else:
                    group_id = np.nan
                data = {
                    'dataset': set_t.name,
                    'sample-id': i,
                    'group-id': group_id,
                    'origin-dataset': set.name,
                    'origin-patient-id': pat_id,
                    'region': region,
                    'empty': empty
                }
                index = append_row(index, data)

    end = time()

    # Write index.
    index = index.astype(cols)
    filepath = os.path.join(set_t.path, 'index.csv')
    index.to_csv(filepath, index=False)

    # Indicate success.
    __write_flag(set_t, f'__CONVERT_FROM_{set.type.name}_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(set_t, hours)

def __destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def __write_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def __print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_{set.type.name}_TIME_HOURS_{hours}__')
    Path(path).touch()

def __create_training_input(
    dataset: 'Dataset',
    index: Union[int, str],
    data: np.ndarray,
    region: Optional[PatientRegion] = None,
    use_compression: bool = True) -> None:
    if region is not None:
        filepath = os.path.join(dataset.path, 'data', 'inputs', region)
    else:
        filepath = os.path.join(dataset.path, 'data', 'inputs')

    if use_compression:
        filepath = os.path.join(filepath, f'{index}.npz')
    else:
        filepath = os.path.join(filepath, f'{index}.np')

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
        
    if use_compression:
        logging.info(f"Saving sample {index}, filepath: {filepath}")
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)

def __create_training_label(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray,
    region: Optional[str] = None,
    use_compression: bool = True) -> None:
    if region is not None:
        filepath = os.path.join(dataset.path, 'data', 'labels', region)
    else:
        filepath = os.path.join(dataset.path, 'data', 'labels')

    if use_compression:
        filepath = os.path.join(filepath, f'{index}.npz')
    else:
        filepath = os.path.join(filepath, f'{index}.np')

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    if use_compression:
        np.savez_compressed(filepath, data=data)
    else:
        np.save(filepath, data)