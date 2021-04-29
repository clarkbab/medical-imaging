import logging
import numpy as np
import os
import pandas as pd
import sys

from .dicom_dataset import DicomDataset

from mymi import config

class HN1(DicomDataset):
    @classmethod
    def data_dir(cls):
        return os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'raw')

    @classmethod
    def clinical_data(cls):
        filepath = os.path.join(config.directories.datasets, 'HEAD-NECK-RADIOMICS-HN1', 'clinical', 'data.csv')
        df = pd.read_csv(filepath)
        df = df.set_index('id')
        return df
