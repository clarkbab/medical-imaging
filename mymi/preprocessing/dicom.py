from distutils.dir_util import copy_tree
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm
import sys
from typing import Optional

from mymi import dataset as ds
from mymi.dataset.processed import recreate as recreate_processed
from mymi.dataset.raw import recreate as recreate_raw
from mymi import logging
from mymi.transforms import centre_crop_or_pad_3D, resample_3D
from mymi import types

FILENAME_NUM_DIGITS = 5

def anonymise(
    dataset: str,
    anon_dataset: str,
    regions: types.PatientRegions = 'all') -> None:

    # Create CT map.
    old_ds = ds.get(dataset)
    pats = old_ds.list_patients(regions=regions)
    map_df = pd.DataFrame(pats, columns=['patient-id'])

    # Save map.
    filename = 'map.csv'
    filepath = os.path.join(old_ds.path, filename)
    map_df.to_csv(filepath)

    # Create new dataset.
    ds = recreate_raw(anon_dataset, type_str='nifti')

    # Add patients to new dataset.
    for anon_id, row in tqdm(map_df.iterrows()):
        # Add CT data.
        data = old_ds.patient(row['patient-id']).ct_data()
        spacing = old_ds.patient(row['patient-id']).ct_spacing()
        offset = old_ds.patient(row['patient-id']).ct_offset()
        affine = np.array([
            [spacing[0], 0, 0, offset[0]],
            [0, spacing[1], 0, offset[1]],
            [0, 0, spacing[2], offset[2]],
            [0, 0, 0, 1]])
        img = Nifti1Image(data, affine)
        filename = f"{anon_id:0{FILENAME_NUM_DIGITS}}.nii.gz"
        filepath = os.path.join(ds.path, 'ct', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        nib.save(img, filepath)

        # Add region data.
        pat_regions = old_ds.patient(row['patient-id']).list_regions()
        regions = np.intersect1d(pat_regions, REGIONS)
        region_data = old_ds.patient(row['patient-id']).region_data(regions=regions)
        for r, d in region_data.items():
            img = Nifti1Image(d.astype(np.int32), affine)
            filename = f"{anon_id:0{FILENAME_NUM_DIGITS}}.nii.gz"
            filepath = os.path.join(ds.path, r, filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            nib.save(img, filepath)

def process(
    dataset: str,
    clear_cache: bool = False,
    dest_dataset: Optional[str] = None,
    p_test: float = 0.2,
    p_train: float = 0.6,
    p_val: float = 0.2,
    random_seed: int = 42,
    regions: types.PatientRegions = 'all',
    size: Optional[types.ImageSize3D] = None,
    spacing: Optional[types.ImageSpacing3D] = None,
    use_mapping: bool = True):
    """
    effect: processes a DICOM dataset and partitions it into train/validation/test
        partitions for training.
    args:
        the dataset to process.
    kwargs:
        clear_cache: force the cache to clear.
        p_test: the proportion of test patients.
        p_train: the proportion of train patients.
        p_val: the proportion of validation patients.
        random_seed: the random seed for shuffling patients.
        regions: the regions to process.
        size: crop/pad to desired size.
        spacing: resample to the desired spacing.
        use_mapping: use region map if present.
    """
    # Load patients.
    old_ds = ds.get(dataset, type_str='dicom')
    pats = old_ds.list_patients(regions=regions) 
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

    # Create dataset.
    proc_ds = recreate_processed(dest_dataset)

    # Save processing params.
    filepath = os.path.join(proc_ds.path, 'params.csv')
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
        proc_ds.create_partition(partition)

        # Write each patient to partition.
        for pat in tqdm(pats):
            # Get available requested regions.
            pat_regions = old_ds.patient(pat).list_regions(use_mapping=use_mapping)

            # Load data.
            input = old_ds.patient(pat).ct_data()
            labels = old_ds.patient(pat).region_data(clear_cache=clear_cache, regions=pat_regions)

            # Resample data if requested.
            if spacing is not None:
                old_spacing = old_ds.patient(pat).ct_spacing()
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
