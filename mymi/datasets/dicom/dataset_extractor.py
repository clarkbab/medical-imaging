import gzip
import numpy as np
import os
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientDataExtractor, PatientInfo

PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed', '2d-parotid-left')

class DatasetExtractor:
    def __init__(self, dataset=ds, verbose=False):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = dataset
        self.verbose = verbose

    def extract(self, drop_missing_slices=True, read_cache=True, regions='all', transforms=[], write_cache=True):
        """
        effect: stores processed patient data.
        drop_missing_slices: drops patients that have missing slices.
        read_cache: read cache entry if present.
        transform: apply the pre-defined transformation.
        write_cache: write to the cache unless read from cache.
        """
        # Load patients.
        pat_ids = self.dataset.list_patients()

        # Maintain global sample index.
        pos_sample_idx = 0
        neg_sample_idx = 0

        # Process data for each patient.
        for pat_id in tqdm(pat_ids):
            if self.verbose:
                print(f"Extracting data for patient {pat_id}.")

            # Check if there are missing slices.
            pi = PatientInfo(pat_id, verbose=self.verbose)
            patient_info_df = pi.full_info(read_cache=read_cache, write_cache=write_cache)

            if drop_missing_slices and patient_info_df['num-missing'][0] != 0:
                if self.verbose: print(f"Dropping patient '{pat_id}' with missing slices.")
                continue

            # Load input data.
            pde = PatientDataExtractor(pat_id, verbose=self.verbose)
            data = pde.get_data(read_cache=read_cache, transforms=transforms, write_cache=write_cache)

            # Load label data.
            labels = pde.get_labels(read_cache=read_cache, regions=regions, transforms=transforms, write_cache=write_cache)

            for lname, ldata in labels:
                # Find slices that are labelled.
                pos_indices = ldata.sum(axis=(0, 1)).nonzero()[0]

                # Write positive input and label data.
                label_path = os.path.join(PROCESSED_ROOT, lname)
                for pos_idx in pos_indices: 
                    # Get input and label data.
                    input_data = data[:, :, pos_idx]
                    label_data = ldata[:, :, pos_idx]

                    # Save input data.
                    pos_path = os.path.join(label_path, 'positive')
                    os.makedirs(pos_path, exist_ok=True)
                    filename = f"{pos_sample_idx:05}-input"
                    filepath = os.path.join(pos_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, input_data)

                    # Save label.
                    filename = f"{pos_sample_idx:05}-label"
                    filepath = os.path.join(pos_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, label_data)

                    # Increment sample index.
                    pos_sample_idx += 1

                # Find slices that aren't labelled.
                neg_indices = np.setdiff1d(range(ldata.shape[2]), pos_indices) 

                # Write negative input and label data.
                for neg_idx in neg_indices:
                    # Get input and label data.
                    input_data = data[:, :, neg_idx]
                    label_data = ldata[:, :, neg_idx]

                    # Save input data.
                    neg_path = os.path.join(label_path, 'negative')
                    os.makedirs(neg_path, exist_ok=True)
                    filename = f"{neg_sample_idx:05}-input"
                    filepath = os.path.join(neg_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, input_data)

                    # Save label.
                    filename = f"{neg_sample_idx:05}-label"
                    filepath = os.path.join(neg_path, filename)
                    f = open(filepath, 'wb')
                    np.save(f, label_data)

                    # Increment sample index.
                    neg_sample_idx += 1
                    



