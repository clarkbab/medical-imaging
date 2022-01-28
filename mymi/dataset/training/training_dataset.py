import numpy as np
import os
import pandas as pd
from typing import Callable, List, Union

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
    def manifest(self) -> pd.DataFrame:
        filepath = os.path.join(self._path, 'manifest.csv')
        df = pd.read_csv(filepath)
        return df

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def patient_id(
        self,
        sample_idx: int) -> types.PatientID:
        df = self.manifest
        result_df = df[df['index'] == sample_idx]
        if len(result_df) == 0:
            raise ValueError(f"Sample '{sample_idx}' not found for dataset '{self}'.")
        pat_id = result_df['patient-id'].iloc[0] 
        return pat_id

    def list_samples(
        self,
        regions: types.PatientRegions = 'all') -> List[int]:
        path = os.path.join(self._path, 'data', 'inputs')
        if os.path.exists(path):
            indices = list(sorted([int(f.replace('.npz', '')) for f in os.listdir(path)]))

            # Filter on sample regions.
            indices = list(filter(self._filter_sample_by_regions(regions), indices))

        else:
            indices = []
        return indices

    def sample(
        self,
        index: Union[int, str],
        by_patient_id: bool = False) -> TrainingSample:
        # Look up sample by patient ID.
        if by_patient_id:
            index = self.manifest[self.manifest['patient-id'] == index].iloc[0]['index']

        return TrainingSample(self, index)

    def _filter_sample_by_regions(
        self,
        regions: types.PatientRegions) -> Callable[[str], bool]:
        """
        returns: a function that filters patients on region presence.
        args:
            regions: the passed 'regions' kwarg.
        """
        def fn(index):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.sample(index).has_region(regions)
            else:
                sam_regions = self.sample(index).list_regions()
                if len(np.intersect1d(regions, sam_regions)) != 0:
                    return True
                else:
                    return False
        return fn
