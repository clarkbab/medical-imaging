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

from mymi import config
from mymi.transforms import resample_3D
from mymi import types

from ...processed import create as create_processed_dataset
from ...processed import destroy as destroy_processed_dataset
from ...processed import list as list_processed_datasets
from .dicom_dataset import DICOMDataset

def process_dicom(
    dataset: str,
    clear_cache: bool = False,
    dest_dataset: Optional[str] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_validation: float = 0.2,
    seed: int = 42,
    regions: types.PatientRegions = 'all',
    spacing: Optional[types.ImageSpacing3D] = None):
    """
    effect: processes a DICOM dataset and partitions it into train/validation/test
        partitions for training.
    args:
        the dataset to process.
    kwargs:
        clear_cache: force the cache to clear.
        p_test: the proportion of test patients.
        p_train: the proportion of train patients.
        p_validation: the proportion of validation patients.
        seed: the random seed for shuffling patients.
        regions: the regions to process.
        spacing: resample to the desired spacing.
    """
    # Load patients who have (at least) one of the required regions.
    ds = DICOMDataset(dataset)
    pats = list(ds.region_names(clear_cache=clear_cache, regions=regions)['patient-id'].unique())
    logging.info(f"Found {len(pats)} patients with (at least) one of the requested regions.")

    # Shuffle patients.
    np.random.seed(seed) 
    np.random.shuffle(pats)

    # Partition patients - rounding assigns more patients to the test set,
    # unless p_test=0, when these are assigned to the validation set.
    num_train = int(np.floor(p_train * len(pats)))
    if p_test == 0:
        num_validation = len(pats) - num_train
    else:
        num_validation = int(np.floor(p_validation * len(pats)))
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
        f.write('key,value\n')
        f.write(f"dataset,{dataset}\n")
        f.write(f"p_test,{p_test}\n")
        f.write(f"p_train,{p_train}\n")
        f.write(f"p_validation,{p_validation}\n")
        f.write(f"seed,{seed}\n")
        f.write(f"regions,'{regions}'\n")
        f.write(f"spacing,'{spacing}'\n")

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
            pat_regions = list(ds.patient(pat).region_names(clear_cache=clear_cache, regions=regions, allow_unknown_regions=True).region)

            # Load data.
            input = ds.patient(pat).ct_data()
            labels = ds.patient(pat).region_data(clear_cache=clear_cache, regions=pat_regions)

            # Resample data if requested.
            if spacing is not None:
                old_spacing = ds.patient(pat).ct_spacing()
                input = resample_3D(input, old_spacing, spacing)
                labels = dict((r, resample_3D(d, old_spacing, spacing)) for r, d in labels.items())

            # Save input data.
            index = proc_ds.partition(partition).create_input(pat, input)

            # Save label data.
            for region, label in labels.items():
                proc_ds.partition(partition).create_label(index, region, label)
