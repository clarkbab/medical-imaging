import logging
import numpy as np
import os
import pandas as pd
import sys

from .dicom_dataset import DicomDataset

data_path = os.environ['MYMI_DATA']
ROOT_DIR = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1')

class HN1(DicomDataset):
    @classmethod
    def data_dir(cls):
        return os.path.join(ROOT_DIR, 'raw')

    @classmethod
    def clinical_data(cls):
        filepath = os.path.join(ROOT_DIR, 'clinical', 'data.csv')
        df = pd.read_csv(filepath)
        df = df.set_index('id')
        return df
