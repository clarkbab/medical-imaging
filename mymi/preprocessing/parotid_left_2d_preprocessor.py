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

PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-2d')

FILENAME_NUM_DIGITS = 5

class ParotidLeft2DPreprocessor:
    def __call__(self, drop_missing_slices=True, num_pats='all', seed=42, transforms=[]):
        """
        effect: stores 2D slice data in 'train', 'validation' and 'test' folders by 
            random split and 'positive' and 'negative' subfolders by presence of 
            'Parotid-Left' gland.
        kwargs:
            drop_missing_slices: drops patients that have missing slices.
            num_pats: operate on subset of patients.
            seed: the random number generator seed.
            transforms: apply the transforms on all patient data.
        """
        # Define label.
        label = 'Parotid-Left'

        # Load patients.
        pat_ids = dataset.list_patients()
        if drop_missing_slices:
            pat_missing_ids = dataset.summary().query('`num-missing` > 0').index.to_numpy()
            pat_ids = np.setdiff1d(pat_ids, pat_missing_ids)
            logging.info(f"Removed {len(pat_missing_ids)} patients with missing slices.")

        # Load patients who have 'Parotid-Left' contours.
        label_df = dataset.labels(pat_id=pat_ids)
        pat_ids = label_df.query(f"`label` == '{label}'")['patient-id'].unique()
        logging.info(f"Found {len(pat_ids)} with '{label}' contours.")

        # Get patient subset for testing.
        if num_pats != 'all':
            assert isinstance(num_pats, int)
            pat_ids = pat_ids[:num_pats]
            logging.info(f"Using subset of {num_pats} patients.")

        # Set split proportions.
        p_train, p_validation, p_test = .6, .2, .2
        logging.info(f"Splitting dataset using proportions: {p_train}/{p_validation}/{p_test}.")

        # Split the patient IDs.
        np.random.seed(seed) 
        np.random.shuffle(pat_ids)
        num_train = int(np.floor(p_train * len(pat_ids)))
        num_validation = int(np.floor(p_validation * len(pat_ids)))
        pat_train_ids = pat_ids[:num_train]
        pat_validation_ids = pat_ids[num_train:(num_train + num_validation)]
        pat_test_ids = pat_ids[(num_train + num_validation):]
        logging.info(f"Num patients in split: {len(pat_train_ids)}/{len(pat_validation_ids)}/{len(pat_test_ids)}.") 

        folders = ['train', 'validation', 'test']
        folder_pat_ids = [pat_train_ids, pat_validation_ids, pat_test_ids]

        # Write data to each folder.
        for folder, pat_ids in zip(folders, folder_pat_ids):
            logging.info(f"Writing data to '{folder}' folder.")

            # Recreate folder.
            folder_path = os.path.join(PROCESSED_ROOT, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

            # Maintain index per subfolder.
            pos_sample_idx = 0
            neg_sample_idx = 0

            # Write each patient to folder.
            for pat_id in tqdm(pat_ids):
                logging.info(f"Extracting data for patient {pat_id}.")

                # Load data.
                data = dataset.patient_data(pat_id, transforms=transforms)
                _, label_data = dataset.patient_labels(pat_id, label=label, transforms=transforms)[0]

                # Find slices with and without 'Parotid-Left' label.
                pos_slice_indices = label_data.sum(axis=(0, 1)).nonzero()[0]
                neg_slice_indices = np.setdiff1d(range(label_data.shape[2]), pos_slice_indices) 

                # Write positive input and label data.
                for pos_slice_i in pos_slice_indices: 
                    # Get input and label data axial slices.
                    d, l = data[:, :, pos_slice_i], label_data[:, :, pos_slice_i]

                    # Save input data.
                    pos_path = os.path.join(folder_path, 'positive')
                    if not os.path.exists(pos_path):
                        os.makedirs(pos_path)
                    filename = f"{pos_sample_idx:0{FILENAME_NUM_DIGITS}}-input"
                    filepath = os.path.join(pos_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, d)

                    # Save label.
                    filename = f"{pos_sample_idx:0{FILENAME_NUM_DIGITS}}-label"
                    filepath = os.path.join(pos_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, l)

                    # Increment sample index.
                    pos_sample_idx += 1

            # Write negative input and label data.
            for neg_slice_i in neg_slice_indices:
                # Get input and label data axial slices.
                d, l = data[:, :, neg_slice_i], label_data[:, :, neg_slice_i]

                # Save input data.
                neg_path = os.path.join(folder_path, 'negative')
                if not os.path.exists(neg_path):
                    os.makedirs(neg_path)
                filename = f"{neg_sample_idx:0{FILENAME_NUM_DIGITS}}-input"
                filepath = os.path.join(neg_path, filename)
                f = open(filepath, 'wb')
                np.save(f, d)

                # Save label.
                filename = f"{neg_sample_idx:0{FILENAME_NUM_DIGITS}}-label"
                filepath = os.path.join(neg_path, filename)
                f = open(filepath, 'wb')
                np.save(f, l)

                # Increment sample index.
                neg_sample_idx += 1
