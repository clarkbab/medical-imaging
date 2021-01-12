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
from mymi.dataset.dicom import PatientDataExtractor, PatientInfo

PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')

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

        # Write volumes for each patient to temporary folder.
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
            data = pde.get_data(transforms=transforms)

            # Load label data.
            labels = pde.get_labels(regions=['Parotid-Left'], transforms=transforms)
            _, label_data = labels[0]

            # Save input volume.
            assert data.shape == label_data.shape
            path = os.path.join(PROCESSED_ROOT, 'temp')
            os.makedirs(path)
            filename = f"{sample_idx:05}-input"
            filepath = os.path.join(path, filename)
            f = open(filepath, 'wb')
            np.save(f, input_data)

            # Save volume label.
            filename = f"{sample_idx:05}-input"
            filepath = os.path.join(path, filename)
            f = open(filepath, 'wb')
            np.save(f, label_data)

            # Increment sample index.
            sample_idx += 1
                