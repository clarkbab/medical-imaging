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

class ParotidLeft2DPreprocessor:
    def extract(self, drop_missing_slices=True, num_pats='all', transforms=[]):
        """
        effect: stores processed patient data.
        kwargs:
            drop_missing_slices: drops patients that have missing slices.
            num_pats: operate on subset of patients.
            transforms: apply the transforms on all patient data.
        """
        # Define region.
        region = 'Parotid-Left'

        # Load patients who have 'Parotid-Left' contours.
        regions_df = dataset.regions()
        pat_ids = regions_df.query(f"`region` == '{region}'")['patient-id'].unique()

        # Get patient subset.
        if num_pats != 'all':
            assert isinstance(num_pats, int)
            pat_ids = pat_ids[:num_pats]

        # Maintain global sample index.
        pos_sample_idx = 0
        neg_sample_idx = 0

        # Write patient CT slices to tmp folder.
        tmp_path = os.path.join(PROCESSED_ROOT, 'tmp')
        os.makedirs(tmp_path, exist_ok=True)
        for pat_id in tqdm(pat_ids):
            logging.info(f"Extracting data for patient {pat_id}.")

            # Check if there are missing slices.
            patient_df = dataset.patient_summary(pat_id)

            if drop_missing_slices and patient_df['num-missing'][0] != 0:
                logging.info(f"Dropping patient '{pat_id}' with missing slices.")
                continue

            # Load input data.
            data = dataset.patient_data(pat_id, transforms=transforms)

            # Load label data.
            _, label_data = dataset.patient_labels(pat_id, regions=region, transforms=transforms)[0]

            # Find slices that are labelled.
            pos_indices = label_data.sum(axis=(0, 1)).nonzero()[0]

            # Write positive input and label data.
            for pos_idx in pos_indices: 
                # Get input and label data.
                input_data = data[:, :, pos_idx]
                ldata = label_data[:, :, pos_idx]

                # Save input data.
                pos_path = os.path.join(tmp_path, 'positive')
                os.makedirs(pos_path, exist_ok=True)
                filename = f"{pos_sample_idx:05}-input"
                filepath = os.path.join(pos_path, filename)
                f = open(filepath, 'wb')
                np.save(f, input_data)

                # Save label.
                filename = f"{pos_sample_idx:05}-label"
                filepath = os.path.join(pos_path, filename)
                f = open(filepath, 'wb')
                np.save(f, ldata)

                # Increment sample index.
                pos_sample_idx += 1

            # Find slices that aren't labelled.
            neg_indices = np.setdiff1d(range(label_data.shape[2]), pos_indices) 

            # Write negative input and label data.
            for neg_idx in neg_indices:
                # Get input and label data.
                input_data = data[:, :, neg_idx]
                ldata = label_data[:, :, neg_idx]

                # Save input data.
                neg_path = os.path.join(tmp_path, 'negative')
                os.makedirs(neg_path, exist_ok=True)
                filename = f"{neg_sample_idx:05}-input"
                filepath = os.path.join(neg_path, filename)
                f = open(filepath, 'wb')
                np.save(f, input_data)

                # Save label.
                filename = f"{neg_sample_idx:05}-label"
                filepath = os.path.join(neg_path, filename)
                f = open(filepath, 'wb')
                np.save(f, ldata)

                # Increment sample index.
                neg_sample_idx += 1

        tmp_folders = ['positive', 'negative']
        new_folders = ['train', 'validate', 'test']

        # Remove processed data from previous run.
        for folder in new_folders:
            folder_path = os.path.join(PROCESSED_ROOT, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(os.path.join(PROCESSED_ROOT, folder))

        # Write shuffled data to new folders.
        for folder in tmp_folders:
            # Shuffle samples for train/validation/test split.
            folder_path = os.path.join(tmp_path, folder)
            samples = np.sort(os.listdir(folder_path)).reshape((-1, 2))
            shuffled_idx = np.random.permutation(len(samples))

            # Get train/test/validate numbers.
            num_train = math.floor(0.6 * len(samples))
            num_validate = math.floor(0.2 * len(samples))
            num_test = math.floor(0.2 * len(samples))
            logging.info(f"Found {len(samples)} samples in folder '{folder}'.")
            logging.info(f"Using train/validate/test split {num_train}/{num_validate}/{num_test}.")

            # Get train/test/validate indices. 
            indices = [shuffled_idx[:num_train], shuffled_idx[num_train:num_validate + num_train], shuffled_idx[num_train + num_validate:num_train + num_validate + num_test]]

            for new_folder, idx in zip(new_folders, indices):
                path = os.path.join(PROCESSED_ROOT, new_folder)
                os.makedirs(os.path.join(path, folder))

                for input_file, label_file in samples[idx]:
                    os.rename(os.path.join(folder_path, input_file), os.path.join(path, folder, input_file))
                    os.rename(os.path.join(folder_path, label_file), os.path.join(path, folder, label_file))

        # Clean up tmp folder.
        shutil.rmtree(tmp_path)
