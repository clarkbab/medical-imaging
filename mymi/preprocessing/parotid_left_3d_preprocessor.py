import gzip
import logging
import math
import numpy as np
import os
from skimage.draw import polygon
import shutil
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi import cache
from mymi import dataset

data_path = os.environ['MYMI_DATA']
PROCESSED_ROOT = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')

FILENAME_NUM_DIGITS = 5

class ParotidLeft3DPreprocessor:
    def extract(self, drop_missing_slices=True, num_pats='all', seed=42, transforms=[]):
        """
        effect: stores 3D patient volumes in 'train', 'validate' and 'test' folders
            by random split.
        kwargs:
            drop_missing_slices: drops patients that have missing slices.
            num_pats: operate on subset of patients.
            seed: the random number generator seed.
            transforms: apply the transforms on all patient data.
        """
        # Define region.
        region = 'Parotid-Left'

        # Load patients.
        pat_ids = dataset.list_patients()
        if drop_missing_slices:
            pat_missing_ids = dataset.summary().query('`num-missing` > 0').index.to_numpy()
            pat_ids = np.setdiff1d(pat_ids, pat_missing_ids)
            logging.info(f"Removed {len(pat_missing_ids)} patients with missing slices.")

        # Load patients who have 'Parotid-Left' contours.
        regions_df = dataset.regions(pat_id=pat_ids)
        pat_ids = regions_df.query(f"`region` == '{region}'")['patient-id'].unique()
        logging.info(f"Found {len(pat_ids)} patients with '{region}' contours.")

        # Get patient subset for testing.
        if num_pats != 'all':
            assert isinstance(num_pats, int)
            pat_ids = pat_ids[:num_pats]
            logging.info(f"Using subset of {num_pats} patients.")

        # Set split proportions.
        p_train, p_validate, p_test = .6, .2, .2
        logging.info(f"Splitting dataset using proportions: {p_train}/{p_validate}/{p_test}.")

        # Split the patient IDs.
        np.random.seed(seed) 
        np.random.shuffle(pat_ids)
        num_train = int(np.floor(p_train * len(pat_ids)))
        num_validate = int(np.floor(p_validate * len(pat_ids)))
        pat_train_ids = pat_ids[:num_train]
        pat_validate_ids = pat_ids[num_train:(num_train + num_validate)]
        pat_test_ids = pat_ids[(num_train + num_validate):]
        logging.info(f"Num patients in split: {len(pat_train_ids)}/{len(pat_validate_ids)}/{len(pat_test_ids)}.") 

        folders = ['train', 'validate', 'test']
        folder_pat_ids = [pat_train_ids, pat_validate_ids, pat_test_ids]

        # Write data to each folder.
        for folder, pat_ids in zip(folders, folder_pat_ids):
            logging.info(f"Writing data to '{folder}' folder.")

            # Recreate folder.
            folder_path = os.path.join(PROCESSED_ROOT, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

            # Maintain index per subfolder.
            sample_idx = 0

            # Write each patient to folder.
            for pat_id in tqdm(pat_ids):
                logging.info(f"Extracting data for patient {pat_id}.")

                # Load data. Don't need to load 'deterministic' transforms as
                # no random transforms should be applied when preprocessing.
                data = dataset.patient_data(pat_id, transforms=transforms)
                _, label_data = dataset.patient_labels(pat_id, regions=region, transforms=transforms)[0]

                # Save input data.
                filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-input"
                filepath = os.path.join(folder_path, filename)
                f = open(filepath, 'wb')
                np.save(f, data)

                # Save label.
                filename = f"{sample_idx:0{FILENAME_NUM_DIGITS}}-label"
                filepath = os.path.join(folder_path, filename)
                f = open(filepath, 'wb')
                np.save(f, label_data)

                # Increment sample index.
                sample_idx += 1
