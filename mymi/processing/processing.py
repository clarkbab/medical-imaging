from datetime import datetime
import matplotlib
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation, label
import shutil
from time import time
from tqdm import tqdm
from typing import *

from mymi.datasets import TrainingDataset
from mymi.datasets.training import create as create_training, exists as exists_training, recreate as recreate_training
from mymi.geometry import extent
from mymi import logging
from mymi.regions import regions_to_list
from mymi.transforms import crop_foreground, resample
from mymi.typing import *
from mymi.utils import *

def convert_brain_crop_to_training(
    set: 'Dataset',
    create_data: bool = True,
    crop_mm: Optional[BoxMM3D] = None,
    dest_dataset: Optional[str] = None,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    load_localiser_prediction: Optional[Callable] = None,
    recreate_dataset: bool = True,
    region: Optional[Regions] = None,
    round_dp: Optional[int] = None,
    spacing: Optional[Spacing3D] = None) -> None:
    logging.arg_log(f'Converting {set.type.name} dataset to TRAINING', ('dataset', 'region'), (set, region))
    regions = regions_to_list(region)

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
    write_flag(set_t, f'__CONVERT_FROM_{set.type.name}_START__')

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
                input = resample(input, spacing=input_spacing, output_spacing=spacing)

            # Crop input.
            if crop_mm is not None:
                # Get crop reference point.
                localiser = ('localiser-Brain', 'public-1gpu-150epochs', 'best')
                brain_label = load_localiser_prediction(set.name, pat_id, localiser)
                if spacing is not None:
                    brain_label = resample(brain_label, spacing=input_spacing, output_spacing=spacing)
                brain_extent = extent(brain_label)
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
                input = crop(input, crop)

            # Save input.
            __create_training_input(set_t, i, input)

            for region in regions:
                # Skip if patient doesn't have region.
                if not set.patient(pat_id).has_regions(region):
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
                    label = resample(label, spacing=input_spacing, output_spacing=spacing)

                # Crop/pad.
                if crop_mm is not None:
                    label = crop(label, crop)

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
    write_flag(set_t, f'__CONVERT_FROM_{set.type.name}_END__')
    hours = int(np.ceil((end - start) / 3600))
    __print_time(set_t, hours)

def __destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def __print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_{set.type.name}_TIME_HOURS_{hours}__')
    Path(path).touch()

def __create_training_input(
    dataset: 'Dataset',
    index: Union[int, str],
    data: np.ndarray,
    region: Optional[Region] = None,
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

def write_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def fill_border_padding(
    data: np.ndarray,
    fill: Union[float, Literal['min']] = 'min') -> np.ndarray:
    size = data.shape
    crop_min = [None] * len(size)
    crop_max = [None] * len(size)
    for axis in range(len(size)):
        # We need to separate this into two opposite passes, as we might get confused
        # by single-valued slices in the middle of the image.
        for slice_idx in range(size[axis]):
            index = [slice(None)] * 3
            index[axis] = slice_idx
            slice_data = data[tuple(index)]
            
            # Look for first slice with more than one intensity value.
            # This is the lower bound (inclusive).
            if len(np.unique(slice_data)) != 1:
                crop_min[axis] = slice_idx
                break
                
        for slice_idx in reversed(range(size[axis])):
            index = [slice(None)] * 3
            index[axis] = slice_idx
            slice_data = data[tuple(index)]
            
            # Look for the first slice with more than one intensity value.
            # The slice above this is the upper bound (if there's a slice above).
            if len(np.unique(slice_data)) != 1:
                crop_max[axis] = slice_idx + 1
                break
                
    crop = (tuple(crop_min), tuple(crop_max)) 
    logging.info(f'Border padding {crop} with values {fill}.')
    data = crop_foreground(data, crop, fill=fill, use_patient_coords=False)
    return data

def fill_contiguous_padding(
    data: np.ndarray,
    fill: Union[float, Literal['min']] = 'min',
    n_largest_intensities: int = 10,
    n_largest_components: int = 10,
    threshold: float = 0.01) -> np.ndarray:
    n_voxels = data.size
    threshold = threshold * n_voxels    # Contiguous objects smaller than this won't be filled.
    fill = data.min() if fill == 'min' else fill

    # Get largest intensity counts.
    def get_largest_value_counts(data: np.ndarray, n: int) -> Tuple[List[float], List[int]]:
        vals, counts = np.unique(data, return_counts=True)
        vals, counts = list(zip(*sorted(zip(vals, counts), key=lambda p: p[1], reverse=True)))
        return vals[:n], counts[:n]
    vals, counts = get_largest_value_counts(data, n=n_largest_intensities)

    for v, _ in zip(vals, counts):
        if v == fill:
            continue

        # Get n largest connected components.
        mask, _ = label(data == v)
        mask_vals, mask_counts = get_largest_value_counts(mask, n=n_largest_components)

        for mv, mc in zip(mask_vals, mask_counts):
            if mv == 0:  # Represents 'background'.
                continue

            # Check size threshold.
            if mc < threshold:
                continue

            # Apply padding.
            logging.info(f'Filling {mc} {v}-valued voxels with {fill} (size={mc / n_voxels:.3f}%)')
            data[mask == mv] = fill

    return data
