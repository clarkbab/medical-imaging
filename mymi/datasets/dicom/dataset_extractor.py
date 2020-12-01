import numpy as np
import os
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientDataExtractor, PatientInfo

PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed')

class DatasetExtractor:
    def __init__(self, dataset=ds, verbose=False):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = dataset
        self.verbose = verbose

    def extract(self, drop_missing=True, read_cache=True, transforms=[], write_cache=True):
        """
        effect: stores processed patient data.
        drop_missing: drops patients that have missing slices.
        read_cache: read cache entry if present.
        transform: apply the pre-defined transformation.
        write_cache: write to the cache unless read from cache.
        """
        # Load patients.
        pat_ids = self.dataset.list_patients()

        # Process data for each patient.
        for pat_id in tqdm(pat_ids):
            # Check if there are missing slices.
            pi = PatientInfo(pat_id, verbose=self.verbose)
            patient_info_df = pi.full_info(read_cache=read_cache, write_cache=write_cache)

            if patient_info_df['num-missing'][0] != 0 and drop_missing:
                if self.verbose: print(f"Dropping patient '{pat_id}' with missing slices.")
                break

            # Process and store input data.
            pde = PatientDataExtractor(pat_id, verbose=self.verbose)
            input_data = pde.get_data(transforms=transforms, read_cache=read_cache, write_cache=write_cache)
            input_path = os.path.join(PROCESSED_ROOT, pat_id, 'input.npy')
            np.save(input_path, input_data)

            # Process and store label data.
            labels = pde.get_labels(read_cache=read_cache, write_cache=write_cache)
            for label_name, label_data in labels:
                ff_label_name = label_name.replace('-', '_').lower()
                label_path = os.path.join(PROCESSED_ROOT, pat_id, 'labels', f"{ff_label_name}.npy")
                np.save(label_path, label_data)
