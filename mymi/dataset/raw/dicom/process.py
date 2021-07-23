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
from mymi import types

from ...processed import create as create_processed_dataset
from ...processed import destroy as destroy_processed_dataset
from ...processed import list as list_processed_datasets
from .dicom_dataset import DICOMDataset

def process_dicom(
    dataset: str,
    include_missing: bool = False,
    dest_dataset: Optional[str] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_validation: float = 0.2,
    random_seed: int = 42,
    regions: types.PatientRegions = 'all'):
    """
    effect: processes a DICOM dataset and partitions it into train/validation/test
        folders for training.
    args:
        the dataset to process.
    kwargs:
        drop_missing: drop patients with missing slices.
        p_test: the proportion of test patients.
        p_train: the proportion of train patients.
        p_validation: the proportion of validation patients.
        regions: the regions to process.
    """
    # Load patients who have (at least) one of the required regions.
    ds = DICOMDataset(dataset)
    pats = list(ds.region_names(regions=regions)['patient-id'].unique())
    logging.info(f"Found {len(pats)} patients with (at least) one of the requested regions.")

    # Drop patients with missing slices.
    if not include_missing:
        pat_ids = list(ds.ct_summary().query('`num-missing` > 0')['patient-id'])
        pats = np.setdiff1d(pats, pat_ids)
        logging.info(f"Removed {len(pat_ids)} patients with missing slices.")

    # Shuffle and partition the patients.
    np.random.seed(random_seed) 
    np.random.shuffle(pats)
    num_train = int(np.floor(p_train * len(pats)))
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

    # Write data to each folder.
    folder_pats = [train_pats, validation_pats, test_pats]
    for folder, pats in zip(proc_ds.folders, folder_pats):
        logging.info(f"Writing '{folder}' patients..")

        # TODO: implement normalisation.

        # Write each patient to folder.
        for pat in tqdm(pats):
            # Get available requested regions.
            pat_regions = list(ds.patient(pat).region_names(regions=regions, allow_unknown_regions=True).region)

            # Load data.
            input = ds.patient(pat).ct_data()
            labels = ds.patient(pat).region_data(regions=pat_regions)

            # Save input data.
            index = proc_ds.create_input(pat, input, folder)

            # Save label data.
            for region, label in labels.items():
                proc_ds.create_label(pat, index, region, label, folder)
