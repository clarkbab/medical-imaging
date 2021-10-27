from distutils.dir_util import copy_tree
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
from scipy.ndimage import binary_dilation
import sys
from typing import Optional

from mymi import logging
from mymi.transforms import centre_crop_or_pad_3D, resample_3D
from mymi import types

from ..nifti import recreate as recreate_nifti
from ..training import recreate as recreate_training

def convert_to_nifti(
    dataset: 'Dataset',
    regions: types.PatientRegions = 'all',
    anonymise: bool = False) -> None:
    # Create NIFTI dataset.
    nifti_ds = recreate_nifti(dataset.name)

    logging.info(f"Converting dataset '{dataset}' to dataset '{nifti_ds}', with regions '{regions}' and anonymise '{anonymise}'.")

    # Load all patients.
    pats = dataset.list_patients(regions=regions)

    if anonymise:
        # Create CT map. Index of map will be the anonymous ID.
        map_df = pd.DataFrame(pats, columns=['patient-id'])

        # Save map.
        filename = 'map.csv'
        filepath = os.path.join(dataset.path, f'anon-nifti-map.csv')
        map_df.to_csv(filepath)

    for pat in tqdm(pats):
        # Get anonymous ID.
        if anonymise:
            anon_id = map_df[map_df['patient-id'] == pat].index.values[0]
            filename = f'{anon_id}.nii.gz'
        else:
            filename = f'{pat}.nii.gz'

        # Create CT NIFTI.
        patient = dataset.patient(pat)
        data = patient.ct_data()
        spacing = patient.ct_spacing()
        offset = patient.ct_offset()
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filepath = os.path.join(nifti_ds.path, 'data', 'ct', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Create region NIFTIs.
        pat_regions = patient.list_regions(whitelist=regions)
        region_data = patient.region_data(regions=pat_regions)
        for region, data in region_data.items():
            img = Nifti1Image(data.astype(np.int32), affine)
            filepath = os.path.join(nifti_ds.path, 'data', region, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

    # Indicate success.
    _indicate_success(nifti_ds, '__CONVERT_TO_NIFTI_SUCCESS__')

def convert_to_training(
    dataset: 'Dataset',
    dest_dataset: str,
    dilate_regions: Optional[types.PatientRegions] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_val: float = 0.2,
    random_seed: int = 42,
    regions: types.PatientRegions = 'all',
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None,
    use_mapping: bool = True):
    # Create dataset.
    train_ds = recreate_training(dest_dataset)

    logging.info(f"Converting dataset '{dataset}' into dataset '{train_ds}', using size '{size}' and spacing '{spacing}'.")

    # Load patients.
    pats = dataset.list_patients(regions=regions) 
    logging.info(f"Found {len(pats)} patients with (at least) one of the requested regions.")

    # Shuffle patients.
    np.random.seed(random_seed) 
    np.random.shuffle(pats)

    # Partition patients - rounding assigns more patients to the test set,
    # unless p_test=0, when these are assigned to the validation set.
    num_train = int(np.floor(p_train * len(pats)))
    if p_test == 0:
        num_validation = len(pats) - num_train
    else:
        num_validation = int(np.floor(p_val * len(pats)))
    train_pats = pats[:num_train]
    validation_pats = pats[num_train:(num_train + num_validation)]
    test_pats = pats[(num_train + num_validation):]
    logging.info(f"Num patients per partition: {len(train_pats)}/{len(validation_pats)}/{len(test_pats)} for train/validation/test.")

    # Save processing params.
    filepath = os.path.join(train_ds.path, 'params.csv')
    with open(filepath, 'w') as f:
        f.write('dataset,p_test,p_train,p_val,random_seed,regions,size,spacing\n')
        f.write(f"{dataset},{p_test},{p_train},{p_val},{random_seed},\"{regions}\",\"{size}\",\"{spacing}\"")

    # Write data to each partition.
    partitions = ['train', 'validation', 'test']
    partition_pats = [train_pats, validation_pats, test_pats]
    for partition, pats in zip(partitions, partition_pats):
        logging.info(f"Writing '{partition}' patients..")

        # TODO: implement normalisation.

        # Create partition.
        train_ds.create_partition(partition)

        # Write each patient to partition.
        for pat in tqdm(pats):
            # Get available requested regions.
            pat_regions = dataset.patient(pat).list_regions(use_mapping=use_mapping, whitelist=regions)

            # Load data.
            input = dataset.patient(pat).ct_data()
            labels = dataset.patient(pat).region_data(regions=pat_regions)

            # Resample data if requested.
            if spacing:
                old_spacing = dataset.patient(pat).ct_spacing()
                input = resample_3D(input, old_spacing, spacing)
                labels = dict((r, resample_3D(d, old_spacing, spacing)) for r, d in labels.items())

            # Crop/pad if requested.
            if size:
                # Log warning if we're cropping the FOV, as we might lose parts of OAR, e.g. Brain.
                fov = np.array(input.shape) * old_spacing
                new_fov = np.array(size) * spacing
                for axis in range(3):
                    if fov[axis] > new_fov[axis]:
                        logging.error(f"Patient FOV larger '{fov}', larger than new FOV '{loc_fov}' for axis '{axis}', losing information.")

                input = centre_crop_or_pad_3D(input, size, fill=np.min(input))
                labels = dict((r, centre_crop_or_pad_3D(d, size, fill=0)) for r, d in labels.items())

            # Dilate the labels if requested.
            if dilate_regions:
                labels = dict((r, binary_dilation(d, iterations=3) if _should_dilate(r, dilate_regions) else d) for r, d in labels.items())

            # Save input data.
            index = train_ds.partition(partition).create_input(pat, input)

            # Save label data.
            for region, label in labels.items():
                train_ds.partition(partition).create_label(index, region, label)

    # Indicate success.
    _indicate_success(nifti_ds, '__CONVERT_TO_TRAINING_SUCCESS__')

def _should_dilate(
    region: str,
    regions: types.PatientRegions) -> bool:
    if type(regions) == str:
        if regions == 'all':
            return True
        elif region == regions:
            return True
    else:
        if region in regions:
            return True
        else:
            return False

def _indicate_success(
    dataset: 'Dataset',
    flag: str) -> None:
    path = os.path.join(dataset.path, flag)
    Path(path).touch()
