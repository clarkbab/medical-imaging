import os

from .processed_dataset import ProcessedDataset

data_path = os.environ['MYMI_DATA']
ROOT_DIR = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1')

class HN1ParotidLeft3D(ProcessedDataset):
    @classmethod
    def data_dir(cls):
        return os.path.join(ROOT_DIR, 'processed', 'parotid-left-3d')


