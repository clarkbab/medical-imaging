import logging
import os

from mymi import dataset
from mymi.dataset.dicom import DicomDataset

class HN1(DicomDataset):
    @classmethod
    def root_dir(cls):
        return os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1', 'raw')

# Set dataset on import for ease of use.
logging.info('Setting HN1 as working dataset.')
dataset.config(dataset=HN1)
