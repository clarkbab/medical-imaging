import gzip
import logging
import math
import numpy as np
import os
import pandas as pd
from skimage.draw import polygon
import shutil
import sys
from torchio import LabelMap, ScalarImage, Subject
from tqdm import tqdm
from typing import *

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import cache
from mymi import config
from mymi import dataset

class ParotidLeft3DPreprocessor:
    def __call__(
        self,
        clear_cache: bool = False,
        drop_missing_slices: bool = True,
        normalise: bool = False, 
        num_pats: Union[str, Sequence[str]] = 'all',
        seed: int = 42,
        transform: bool = None):
        """
        effect: stores 3D patient volumes in 'train', 'validate' and 'test' folders
            by random split.
        kwargs:
            clear_cache: force the cache to clear.
            drop_missing_slices: drops patients that have missing slices.
            normalise: normalise the input data.
            num_pats: operate on subset of patients.
            seed: the random number generator seed.
            transforms: apply the transforms on all patient data.
        """
        # Define regions.
        regions = 'Parotid-Left'

        # Load patients.
        pats = dataset.list_patients()

        # Drop patients with missing slices.
        if drop_missing_slices:
            pat_missing_ids = dataset.ct_summary(clear_cache=clear_cache).query('`num-missing` > 0').index.to_numpy()
            pats = np.setdiff1d(pats, pat_missing_ids)
            logging.info(f"Removed {len(pat_missing_ids)} patients with missing slices.")

        # Load patients who have 'Parotid-Left' contours.
        regions_df = dataset.region_summary(clear_cache=clear_cache, regions=regions, pat_ids=pats)
        pats = regions_df['patient-id'].unique()
        logging.info(f"Found {len(pats)} patients with one of '{regions}' regions.")

        # Get patient subset for testing.
        if num_pats != 'all':
            assert isinstance(num_pats, int)
            pats = pats[:num_pats]
            logging.info(f"Using subset of {num_pats} patients.")

        # Set split proportions.
        p_train, p_validate, p_test = .6, .2, .2
        logging.info(f"Splitting dataset using proportions: {p_train}/{p_validate}/{p_test}.")

        # Split the patient IDs.
        np.random.seed(seed) 
        np.random.shuffle(pats)
        num_train = int(np.floor(p_train * len(pats)))
        num_validate = int(np.floor(p_validate * len(pats)))
        train_pats = pats[:num_train]
        validate_pats = pats[num_train:(num_train + num_validate)]
        test_pats = pats[(num_train + num_validate):]
        logging.info(f"Num patients in split: {len(train_pats)}/{len(validate_pats)}/{len(test_pats)}.") 

        # Write data to each folder.
        folders = ['train', 'validate', 'test']
        folder_pats = [train_pats, validate_pats, test_pats]
        for folder, pats in zip(folders, folder_pats):
            logging.info(f"Writing data to '{folder}' folder.")

            # Recreate folder.
            folder_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

            # Load dataset statistics for normalisation.
            if normalise:
                stats_df = dataset.data_statistics(clear_cache=clear_cache, region=region)
                mean, std_dev = stats_df['hu-mean'].item(), stats_df['hu-std-dev'].item() 

            # Maintain index per subfolder.
            sample_idx = 0

            # Write patient manifest.
            logging.info(f"Writing {folder} manifest file.")
            manifest_path = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'manifests', f"{folder}.csv")
            os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
            df = pd.DataFrame(pats, columns=['patient-id'])
            df.to_csv(manifest_path, index=False) 

            # Write each patient to folder.
            for pat in tqdm(pats):
                logging.info(f"Extracting data for patient {pat}.")

                # Load data.
                # Terminology: 'regions' become 'labels' when the data is prepared for model training.
                input = dataset.patient(pat).ct_data(clear_cache=clear_cache)
                label_data = dataset.patient(pat).region_data(clear_cache=clear_cache, regions=regions)['Parotid-Left']

                # Normalise data.
                if normalise:
                    input = (input - mean) / std_dev

                # Perform transform.
                if transform:
                    # Load patient summary.
                    summary = dataset.patient(pat).ct_summary(clear_cache=clear_cache).iloc[0].to_dict()

                    # Add 'batch' dimension.
                    input = np.expand_dims(input, axis=0)
                    label_data = np.expand_dims(label_data, axis=0)

                    # Create 'subject'.
                    affine = np.array([
                        [summary['spacing-x'], 0, 0, 0],
                        [0, summary['spacing-y'], 0, 0],
                        [0, 0, summary['spacing-z'], 1],
                        [0, 0, 0, 1]
                    ])
                    input = ScalarImage(tensor=input, affine=affine)
                    label_data = LabelMap(tensor=label_data, affine=affine)
                    subject = Subject(input=input, label=label_data)

                    # Transform the subject.
                    output = transform(subject)

                    # Extract results.
                    input = output['input'].data.squeeze(0)
                    label_data = output['label'].data.squeeze(0)

                # Save input data.
                filename = f"{sample_idx:0{config.formatting.sample_digits}}-input"
                filepath = os.path.join(folder_path, filename)
                f = open(filepath, 'wb')
                np.save(f, input)

                # Save label.
                filename = f"{sample_idx:0{config.formatting.sample_digits}}-label"
                filepath = os.path.join(folder_path, filename)
                f = open(filepath, 'wb')
                np.save(f, label_data)

                # Increment sample index.
                sample_idx += 1
