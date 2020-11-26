import numpy as np
import os
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.datasets.dicom import DicomDataset as ds
from mymi.datasets.dicom import PatientDataExtractor

PROCESSED_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'processed')

class DatasetExtractor:
    def __init__(self, dataset=ds):
        """
        dataset: a DicomDataset object.
        """
        self.dataset = dataset

    def extract(self):
        """
        effect: stores processed patient data.
        """
        # Load patients.
        pat_ids = self.dataset.list_patients()

        # Process data for each patient.
        for pat_id in tqdm(pat_ids):
            # Process and store input data.
            pde = PatientDataExtractor.from_id(pat_id, dataset=self.dataset)
            input_data = pde.get_data()
            input_path = os.path.join(PROCESSED_ROOT, pat_id, 'input.npy')
            np.save(input_path, input_data)

            # Process and store label data.
            labels = pde.list_labels()
            for label_name, label_data in labels:
                ff_label_name = label_name.replace('-', '_').lower()
                label_path = os.path.join(PROCESSED_ROOT, pat_id, 'labels', f"{ff_label_name}.npy")
                np.save(label_path, label_data)
