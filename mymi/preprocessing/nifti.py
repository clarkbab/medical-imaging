import logging
import numpy as np
import os
import pandas as pd
from skimage.draw import polygon
import sys
from torchio import LabelMap, ScalarImage, Subject
from tqdm import tqdm
from typing import Optional

from mymi.dataset.processed import recreate as recreate_processed
from mymi.dataset.raw import recreate as recreate_raw
from mymi.transforms import centre_crop_or_pad_3D, resample_3D
from mymi import types

def anonymise(
    dataset: str,
    anon_dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> None:
    # Copy dataset.
    old_ds = dataset.get(dataset, type_str='nifti')
    new_ds = recreate_raw(anon_dataset, type_str='nifti')
    logging.info('Copying dataset...')
    copy_tree(old_ds.path, new_ds.path)
    logging.info('Copied.')

    # Create CT map.
    ct_path = os.path.join(new_ds.path, 'ct')
    ct_files = list(sorted(os.listdir(ct_path)))
    ct_ids = list(f.replace('.nii.gz', '') for f in ct_files)
    map_df = pd.DataFrame(ct_ids, columns=['patient-id'])

    # Save map.
    filepath = os.path.join(old_ds_path, 'map.csv')
    map_df.to_csv(filepath)

    for anon_id, row in tqdm(map_df.iterrows()):
        # Rename CT files.
        old_filename = f"{row['patient-id']}.nii.gz" 
        old_filepath = os.path.join(new_ds.path, 'ct', old_filename)
        new_filename = f"{anon_id:0{FILENAME_NUM_DIGITS}}.nii.gz"
        new_filepath = os.path.join(new_ds.path, 'ct', new_filename)
        shutil.move(old_filepath, new_filepath)

        # Rename region files.
        for region in REGIONS:
            old_filename = f"{row['patient-id']}.nii.gz" 
            old_filepath = os.path.join(new_ds.path, region, old_filename)
            new_filename = f"{anon_id:0{FILENAME_NUM_DIGITS}}.nii.gz"
            new_filepath = os.path.join(new_ds.path, region, new_filename)
            if os.path.exists(old_filepath):
                shutil.move(old_filepath, new_filepath)

def process(
    dataset: str,
    dest_dataset: str,
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
        dataset: the dataset to process.
        dest_dataset: the processed dataset.
    kwargs:
        p_test: the proportion of test patients.
        p_train: the proportion of train patients.
        p_val: the proportion of validation patients.
        regions: the regions to process.
        size: crop/pad to the desired size.
        spacing: resample to the desired spacing.
    """
    logging.info(f"Processing '{dataset}' dataset into '{dest_dataset}' dataset.")

    # Load patients.
    old_ds = ds.get(dataset, type_str='nifti')
    pats = old_ds.list_patients()
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

    # Create dataset.
    proc_ds = recreate_processed(name)

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
            pat_regions = old_ds.patient(pat).list_regions()

            # Load data.
            input = old_ds.patient(pat).ct_data()
            labels = old_ds.patient(pat).region_data(regions=pat_regions)

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
