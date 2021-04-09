import os

from mymi import config

from .processed_dataset import ProcessedDataset

class HN1ParotidLeft3D(ProcessedDataset):
    @classmethod
    def data_dir(cls):
        return os.path.join(config.dataset_dir, 'HEAD-NECK-RADIOMICS-HN1', 'processed', 'parotid-left-3d')


