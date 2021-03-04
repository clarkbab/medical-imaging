import logging
import numpy as np
import os
import pandas as pd

from mymi import dataset
from mymi.dataset import Dataset

data_path = os.environ['MYMI_DATA']
ROOT_DIR = os.path.join(data_path, 'datasets', 'HEAD-NECK-RADIOMICS-HN1')

class HN1(Dataset):
    @classmethod
    def data_dir(cls):
        return os.path.join(ROOT_DIR, 'raw')

    @classmethod
    def clinical_data(cls):
        filepath = os.path.join(ROOT_DIR, 'clinical', 'data.csv')
        df = pd.read_csv(filepath)
        df = df.set_index('id')
        return df

# Set dataset on import for ease of use.
logging.info('Setting HN1 as working dataset.')
dataset.config(dataset=HN1)
