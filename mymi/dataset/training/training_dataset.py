import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional, Union

from mymi import config
from mymi import types

from ..dataset import Dataset, DatasetType
from .training_sample import TrainingSample

class TrainingDataset(Dataset):
    def __init__(
        self,
        name: str,
        check_conversion: bool = True):
        """
        args:
            name: the name of the dataset.
        """
        self._global_id = f"TRAINING: {name}"
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'training', name)

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self}' not found.")

        # Check if converted successfully.
        if check_conversion:
            path = os.path.join(self._path, '__CONVERT_FROM_NIFTI_START__')
            if os.path.exists(path):
                path = os.path.join(self._path, '__CONVERT_FROM_NIFTI_END__')
                if not os.path.exists(path):
                    raise ValueError(f"Dataset '{self}' conversion isn't finished.")

        # Load data index.
        filepath = os.path.join(self._path, 'index.csv')
        self._index = pd.read_csv(filepath, dtype={ 'sample-id': str })

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def params(self) -> pd.DataFrame:
        filepath = os.path.join(self._path, 'params.csv')
        df = pd.read_csv(filepath)
        params = df.iloc[0].to_dict()
        
        # Replace special columns.
        cols = ['size', 'spacing']
        for col in cols:
            if col == 'None':
                params[col] = None
            else:
                params[col] = eval(params[col])
        return params

    @property
    def index(self) -> pd.DataFrame:
        filepath = os.path.join(self._path, 'index.csv')
        df = pd.read_csv(filepath)
        return df

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def patient_id(
        self,
        sample_idx: int) -> types.PatientID:
        df = self._index[self._index['sample-id'] == sample_idx]
        if len(df) == 0:
            raise ValueError(f"Sample '{sample_idx}' not found for dataset '{self}'.")
        pat_id = df['patient-id'].iloc[0] 
        return pat_id

    def list_samples(
        self,
        regions: Optional[Union[str, List[str]]] = None) -> List[int]:
        if type(regions) == str:
            regions = [regions]

        index = self._index

        # Filter by regions.
        if regions is not None:
            index = index[index.region.isin(regions)]

        sample_ids = list(sorted(index['sample-id'].unique()))
        return sample_ids

    def sample(
        self,
        sample_id: Union[int, str],
        by_patient_id: bool = False) -> TrainingSample:
        # Look up sample by patient ID.
        if by_patient_id:
            sample_id = self._index[self._index['patient-id'] == sample_id].iloc[0]['sample-id']

        return TrainingSample(self, sample_id)
