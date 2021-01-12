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
from mymi.dataset.dicom import PatientDataExtractor, PatientInfo, DatasetInfo

PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '3d-parotid-left')

class DatasetPreprocessor3D:
    def extract(self, drop_missing_slices=True, num_pats='all', transforms=[]):
        """
        effect: stores patient volumes in a format ready for training.
        drop_missing_slices: drop patients who have missing slices.
        num_pats: operate on a subset of patients for testing.
        transforms: apply the transforms to the patient volume.
        """
        # Load patients with 'Parotid-Left' contours.
        info = DatasetInfo() 
        regions = info.patient_regions()
        pat_ids = regions.query("`region` == 'Parotid-Left'")['patient-id'].unique()

        # Get patient subset.
        if num_pats != 'all':
            assert isinstance(num_pats, int)
            pat_ids = pat_ids[:num_pats]

        # Maintain global sample index.
        sample_idx = 0

        # Write patient CT volumes to tmp folder.
        tmp_path = os.path.join(PROCESSED_ROOT, 'tmp')
        os.makedirs(tmp_path, exist_ok=True)
        for pat_id in tqdm(pat_ids):
            logging.info(f"Extracting data for patient {pat_id}.")

            # Check if there are missing slices.
            info = PatientInfo(pat_id)
            patient_info_df = info.full_info()

            if drop_missing_slices and patient_info_df['num-missing'][0] != 0:
                logging.info(f"Dropping patient '{pat_id}' with missing slices.")
                continue

            # Load input data.
            pde = PatientDataExtractor(pat_id)
            input_data = pde.get_data(transforms=transforms)

            # Load label data.
            labels = pde.get_labels(regions=['Parotid-Left'], transforms=transforms)
            _, label_data = labels[0]

            # Save input volume.
            assert input_data.shape == label_data.shape
            filename = f"{sample_idx:05}-input"
            filepath = os.path.join(tmp_path, filename)
            f = open(filepath, 'wb')
            np.save(f, input_data)

            # Save volume label.
            filename = f"{sample_idx:05}-label"
            filepath = os.path.join(tmp_path, filename)
            f = open(filepath, 'wb')
            np.save(f, label_data)

            # Increment sample index.
            sample_idx += 1

        # Shuffle volumes and write to train, validate and test folders.
        new_folders = ['train', 'validate', 'test']
                
        # Remove processed data from previous run.
        for folder in new_folders:
            folder_path = os.path.join(PROCESSED_ROOT, folder)
            if os.path.exists(folder_path):
                shutil.rmtree(os.path.join(PROCESSED_ROOT, folder))

        # Shuffle samples.
        samples = np.sort(os.listdir(tmp_path)).reshape((-1, 2))
        shuffled_idx = np.random.permutation(len(samples))

        num_train = math.floor(0.6 * len(samples))
        num_validate = math.floor(0.2 * len(samples))
        num_test = math.floor(0.2 * len(samples))
        logging.info(f"Found {len(samples)} samples in folder '{folder}'.")
        logging.info(f"Using train/validate/test split {num_train}/{num_validate}/{num_test}.")

        # Get train/validate/test indices. 
        indices = [shuffled_idx[:num_train], shuffled_idx[num_train:num_validate + num_train], shuffled_idx[num_train + num_validate:num_train + num_validate + num_test]]

        # Copy data to new folders.
        for new_folder, idx in zip(new_folders, indices):
            folder_path = os.path.join(PROCESSED_ROOT, new_folder)
            os.makedirs(folder_path)

            for input_file, label_file in samples[idx]:
                os.rename(os.path.join(tmp_path, input_file), os.path.join(folder_path, input_file))
                os.rename(os.path.join(tmp_path, label_file), os.path.join(folder_path, label_file))

        # Clean up tmp folder.
        shutil.rmtree(tmp_path)
