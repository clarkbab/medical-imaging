import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.ndimage import binary_dilation
from time import time
from tqdm import tqdm
from typing import List, Optional

from mymi import types
from mymi.dataset.nifti import NIFTIDataset
from mymi.dataset.training import recreate as recreate_training
from mymi import logging
from mymi.transforms import resample_3D, top_crop_or_pad_3D

def convert_to_training(
    dataset: str,
    regions: List[str],
    dest_dataset: str,
    dilate_regions: List[str] = [],
    round_dp: Optional[int] = None,
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None) -> None:

    # Create the dataset.
    train_ds = recreate_training(dest_dataset)
    _write_flag(train_ds, '__CONVERT_FROM_NIFTI_START__')

    # Notify user.
    logging.info(f"Creating dataset '{train_ds}' with regions={regions}, dilate_regions={dilate_regions}, size={size} and spacing={spacing}.")

    # Save processing params.
    filepath = os.path.join(train_ds.path, 'params.csv')
    params_df = pd.DataFrame({
        'dilate-regions': [str(regions)],
        'regions': [str(dilate_regions)],
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
        old_spacing = patient.ct_spacing()
        input = patient.ct_data()

        # Resample input.
        if spacing:
            input = resample_3D(input, old_spacing, spacing)

        # Save input.
        _create_training_input(train_ds, i, input)

        # Add to manifest.
        _append_to_manifest(set, train_ds, pat, i)

        # Get regions.
        regions = set.patient(pat).list_regions()

        for region in regions:
            # Load label data.
            label = patient.region_data(regions=region)[region]

            # Resample data.
            if spacing:
                label = resample_3D(label, old_spacing, spacing)

            # Crop/pad.
            if size:
                # Log warning if we're cropping the FOV, as we might lose parts of OAR, e.g. Brain.
                fov = np.array(input.shape) * spacing
                new_fov = np.array(size) * spacing
                for axis in range(len(size)):
                    if fov[axis] > new_fov[axis]:
                        logging.error(f"Patient FOV '{fov}', larger than new FOV '{new_fov}' for axis '{axis}', losing information for patient '{patient}'.")
                input = top_crop_or_pad_3D(input, size, fill=np.min(input))
                label = top_crop_or_pad_3D(label, size, fill=0)

            # Round data after resampling to save on disk space.
            if round_dp is not None:
                input = np.around(input, decimals=round_dp)

            # Dilate the labels if requested.
            if region in dilate_regions:
                label = binary_dilation(label, iterations=3)

            # Save label. Filter out labels with no foreground voxels, e.g. from resampling small OARs.
            if label.sum() != 0:
                _create_training_label(train_ds, i, region, label)

    end = time()

    # Indicate success.
    _write_flag(train_ds, '__CONVERT_FROM_NIFTI_END__')
    hours = int((end - start) / 3600)
    _print_time(train_ds, hours)

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
