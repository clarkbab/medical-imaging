import os

from .dicom_dataset import DicomDataset

class Test(DicomDataset):
    @classmethod
    def data_dir(cls):
        return os.path.join('this', 'is', 'it')

    @classmethod
    def clinical_data(cls):
        return None
