import logging
import numpy as np
import os
import pandas as pd
from skimage.draw import polygon
import sys
from torchio import LabelMap, ScalarImage, Subject
from tqdm import tqdm
from typing import Optional

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.transforms import centre_crop_or_pad_3D, resample_3D
from mymi import types

from ...processed import create as create_processed_dataset
from ...processed import destroy as destroy_processed_dataset
from ...processed import list as list_processed_datasets
from .nifti_dataset import NIFTIDataset

def process_nifti(
    dataset: str,
    dest_dataset: Optional[str] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_val: float = 0.2,
    seed: int = 42,
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None):
    """
    effect: processes a NIFTI dataset and partitions it into train/validation/test
        partitions for training.
    args:
        the dataset to process.
    kwargs:
        p_test: the proportion of test patients.
        p_train: the proportion of train patients.
        p_val: the proportion of validation patients.
        regions: the regions to process.
        size: crop/pad to the desired size.
        spacing: resample to the desired spacing.
    """
    # Load patients.
    ds = NIFTIDataset(dataset)
    pats = ds.list_patients()
    logging.info(f"Found {len(pats)} patients.")

    # Shuffle patients.
    np.random.seed(seed) 
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

    # Destroy old dataset if present.
    name = dest_dataset if dest_dataset else dataset
    if name in list_processed_datasets():
        destroy_processed_dataset(name)

    # Create dataset.
    proc_ds = create_processed_dataset(name)

    # Save processing params.
    filepath = os.path.join(proc_ds.path, 'params.csv')
    with open(filepath, 'w') as f:
        f.write('dataset,p_test,p_train,p_val,seed,size,spacing\n')
        f.write(f"{dataset},{p_test},{p_train},{p_val},{seed},\"{size}\",\"{spacing}\"")

    # Write data to each partition.
    partitions = ['train', 'validation', 'test']
    partition_pats = [train_pats, validation_pats, test_pats]
    for partition, pats in zip(partitions, partition_pats):
        logging.info(f"Writing '{partition}' patients..")

        # TODO: implement normalisation.

        # Create partition.
        proc_ds.create_partition(partition)

        # Write each patient to partition.
        for pat in tqdm(pats):
            # Get available requested regions.
            pat_regions = ds.patient(pat).list_regions()

            # Load data.
            input = ds.patient(pat).ct_data()
            labels = ds.patient(pat).region_data(regions=pat_regions)

            # Resample data if requested.
            if spacing is not None:
                old_spacing = ds.patient(pat).ct_spacing()
                input = resample_3D(input, old_spacing, spacing)
                labels = dict((r, resample_3D(d, old_spacing, spacing)) for r, d in labels.items())

            # Crop/pad if requested.
            if size is not None:
                input = centre_crop_or_pad_3D(input, size, fill=np.min(input))
                labels = dict((r, centre_crop_or_pad_3D(d, size, fill=0)) for r, d in labels.items())

            # Save input data.
            index = proc_ds.partition(partition).create_input(pat, input)

            # Save label data.
            for region, label in labels.items():
                proc_ds.partition(partition).create_label(index, region, label)
