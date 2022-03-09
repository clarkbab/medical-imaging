import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation
import shutil
from time import time
from tqdm import tqdm
from typing import List, Optional, Union

from mymi import types
from mymi.dataset.nifti import NIFTIDataset
from mymi.dataset.training import create, exists, get, recreate
from mymi import logging
from mymi.transforms import resample_3D, top_crop_or_pad_3D

def convert_to_training(
    dataset: str,
    regions: Union[str, List[str]],
    dest_dataset: str,
    dilate_iter: int = 3,
    dilate_regions: List[str] = [],
    log_warnings: bool = False,
    recreate_dataset: bool = True,
    round_dp: Optional[int] = None,
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None) -> None:
    if type(regions) == str:
        regions = [regions]

    # Create the dataset.
    if exists(dest_dataset):
        if recreate_dataset:
            created = True
            train_ds = recreate(dest_dataset)
        else:
            created = False
            train_ds = get(dest_dataset)
            _destroy_flag(train_ds, '__CONVERT_FROM_NIFTI_END__')

            # Delete old labels.
            for region in regions:
                filepath = os.path.join(train_ds.path, 'data', 'labels', region)
                shutil.rmtree(filepath)
    else:
        created = True
        train_ds = create(dest_dataset)
    _write_flag(train_ds, '__CONVERT_FROM_NIFTI_START__')

    # Notify user.
    logging.info(f"Creating dataset '{train_ds}' with recreate_dataset={recreate_dataset}, regions={regions}, dilate_regions={dilate_regions}, dilate_iter={dilate_iter}, size={size} and spacing={spacing}.")

    # Save processing params.
    if created:
        filepath = os.path.join(train_ds.path, 'params.csv')
        params_df = pd.DataFrame({
            'dilate-iter': [str(dilate_iter)],
            'dilate-regions': [str(dilate_regions)],
            'regions': [str(regions)],
            'size': [str(size)] if size is not None else ['None'],
            'spacing': [str(spacing)] if spacing is not None else ['None'],
        })
        params_df.to_csv(filepath)
    else:
        for region in regions:
            filepath = os.path.join(train_ds.path, f'params-{region}.csv')
            params_df = pd.DataFrame({
                'dilate-iter': [str(dilate_iter)],
                'dilate-regions': [str(dilate_regions)],
                'regions': [str(regions)],
                'size': [str(size)] if size is not None else ['None'],
                'spacing': [str(spacing)] if spacing is not None else ['None'],
            })
            params_df.to_csv(filepath)

    # Load patients.
    set = NIFTIDataset(dataset)
    pats = set.list_patients()

    # Write each patient to dataset.
    start = time()
    for i, pat in enumerate(tqdm(pats)):
        # Load input data.
        patient = set.patient(pat)
        old_spacing = patient.ct_spacing
        input = patient.ct_data

        # Resample input.
        if spacing:
            input = resample_3D(input, old_spacing, spacing)

        # Crop/pad.
        if size:
            # Log warning if we're cropping the FOV as we're losing information.
            if log_warnings:
                if spacing:
                    fov_spacing = spacing
                else:
                    fov_spacing = old_spacing
                fov = np.array(input.shape) * fov_spacing
                new_fov = np.array(size) * fov_spacing
                for axis in range(len(size)):
                    if fov[axis] > new_fov[axis]:
                        logging.warning(f"Patient '{patient}' had FOV '{fov}', larger than new FOV after crop/pad '{new_fov}' for axis '{axis}'.")

            # Perform crop/pad.
            input = top_crop_or_pad_3D(input, size, fill=input.min())

        # Save input.
        _create_training_input(train_ds, i, input)

        # Add to manifest.
        _append_to_manifest(set, train_ds, pat, i)

        for region in regions:
            # Skip if patient doesn't have region.
            if not set.patient(pat).has_region(region):
                continue

            # Load label data.
            label = patient.region_data(regions=region)[region]

            # Resample data.
            if spacing:
                label = resample_3D(label, old_spacing, spacing)

            # Crop/pad.
            if size:
                label = top_crop_or_pad_3D(label, size)

            # Round data after resampling to save on disk space.
            if round_dp is not None:
                input = np.around(input, decimals=round_dp)

            # Dilate the labels if requested.
            if region in dilate_regions:
                label = binary_dilation(label, iterations=dilate_iter)

            # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
            if label.sum() != 0:
                _create_training_label(train_ds, i, region, label)

    end = time()

    # Indicate success.
    _write_flag(train_ds, '__CONVERT_FROM_NIFTI_END__')
    hours = int(np.ceil((end - start) / 3600))
    _print_time(train_ds, hours)

def _destroy_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    os.remove(path)

def _write_flag(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()

def _print_time(
    dataset: 'Dataset',
    hours: int) -> None:
    path = os.path.join(dataset.path, f'__CONVERT_FROM_NIFTI_TIME_HOURS_{hours}__')
    Path(path).touch()

def _create_training_input(
    dataset: 'Dataset',
    index: int,
    data: np.ndarray) -> None:
    # Save the input data.
    filepath = os.path.join(dataset.path, 'data', 'inputs', f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)

def _append_to_manifest(
    source_dataset: 'Dataset',
    dataset: 'Dataset',
    patient: str,
    index: int) -> None:
    # Append to manifest.
    manifest_path = os.path.join(dataset.path, 'manifest.csv')
    if not os.path.exists(manifest_path):
        with open(manifest_path, 'w') as f:
            f.write('dataset,patient-id,dest-dataset,index\n')

    # Append line to manifest. 
    with open(manifest_path, 'a') as f:
        f.write(f"{source_dataset.name},{patient},{dataset.name},{index}\n")

def _create_training_label(
    dataset: 'Dataset',
    index: int,
    region: str,
    data: np.ndarray) -> None:
    # Save the label data.
    filepath = os.path.join(dataset.path, 'data', 'labels', region, f'{index}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=data)
