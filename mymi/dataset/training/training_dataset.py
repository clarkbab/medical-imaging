import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional, Union

from mymi import config
from mymi.types import PatientRegions
from mymi.utils import arg_to_list

from ..dataset import Dataset, DatasetType
from .training_sample import TrainingSample

class TrainingDataset(Dataset):
    def __init__(
        self,
        name: str,
        check_processed: bool = True,
        **kwargs):
        self.__index = None     # Lazy-loaded.
        self.__name = name
        self.__global_id = f"TRAINING: {self.__name}"
        self.__path = os.path.join(config.directories.datasets, 'training', self.__name)
        self.__loader_split = None     # Lazy-loaded.

        # Check if dataset exists.
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

        # Check if processing from NIFTI/NRRD has completed.
        if check_processed:
            path = os.path.join(self.__path, '__CONVERT_FROM_NIFTI_START__')
            if os.path.exists(path):
                path = os.path.join(self.__path, '__CONVERT_FROM_NIFTI_END__')
                if not os.path.exists(path):
                    raise ValueError(f"Dataset '{self}' processing from NIFTI not completed. To override check use 'check_processed=False'.")
            path = os.path.join(self.__path, '__CONVERT_FROM_NRRD_START__')
            if os.path.exists(path):
                path = os.path.join(self.__path, '__CONVERT_FROM_NRRD_END__')
                if not os.path.exists(path):
                    raise ValueError(f"Dataset '{self}' processing from NRRD not completed. To override check use 'check_processed=False'.")

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def has_grouping(self) -> bool:
        return 'group-id' in self.index

    @property
    def index(self) -> str:
        if self.__index is None:
            self.__load_index()
        return self.__index

    @property
    def loader_split(self) -> pd.DataFrame:
        if self.__loader_split is None:
            self.__load_loader_split()
            self.__loaded_loader_split = True
        return self.__loader_split

    @property
    def name(self) -> str:
        return self.__name

    @property
    def params(self) -> pd.DataFrame:
        filepath = os.path.join(self.__path, 'params.csv')
        df = pd.read_csv(filepath)
        params = df.iloc[0].to_dict()
        params['regions'] = eval(params['regions'])
        if 'spacing' in params:
            params['spacing'] = eval(params['spacing'])
        elif 'output-spacing' in params:
            params['output-spacing'] = eval(params['output-spacing'])
        return params

    @property
    def path(self) -> str:
        return self.__path

    @property
    def type(self) -> DatasetType:
        return DatasetType.TRAINING

    def list_groups(
        self,
        include_empty: bool = False,
        region: Optional[PatientRegions] = None,
        sort_by_sample_id: bool = False) -> List[int]:
        if not self.has_grouping:
            raise ValueError(f"{self} has no grouping.")

        # Filter out 'empty' labels.
        index = self.index
        if 'empty' in self.index.columns and not include_empty:
            index = index[~index['empty']]

        # Filter by regions.
        if region is not None:
            regions = arg_to_list(region, str)
            index = index[index['region'].isin(regions)]

        # Get sample IDs.
        if sort_by_sample_id:
            group_ids = list(index.sort_values('sample-id')['group-id'].unique())
        else:
            group_ids = list(sorted(index['group-id'].unique()))
        group_ids = [int(i) for i in group_ids]

        return group_ids

    def list_regions(self) -> List[str]:
        return list(sorted(self.index['region'].unique())) 

    def list_samples(
        self,
        group_id: Optional[Union[int, List[int]]] = None,
        include_empty: bool = False,
        region: Optional[PatientRegions] = None) -> List[int]:
        group_ids = arg_to_list(group_id, int)

        # Filter out 'empty' labels.
        index = self.index
        if 'empty' in self.index.columns and not include_empty:
            index = index[~index['empty']]

        # Filter by regions.
        if region is not None:
            regions = arg_to_list(region, str)
            index = index[index['region'].isin(regions)]

        # Filter by groups.
        if group_id is not None:
            index = index[index['group-id'].isin(group_ids)] 

        # Get sample IDs.
        sample_ids = list(sorted(index['sample-id'].unique()))
        sample_ids = [int(i) for i in sample_ids]

        return sample_ids

    def sample(
        self,
        sample_id: Union[int, str],
        by_origin_id: bool = False) -> TrainingSample:
        # Look up sample by patient ID.
        if by_origin_id:
            sample_id = self.__index[self.__index['origin-patient-id'] == sample_id].iloc[0]['sample-id']
        return TrainingSample(self, sample_id)

    def __load_index(self) -> None:
        filepath = os.path.join(self.__path, 'index.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"Index not found for {self}.")
        self.__index = pd.read_csv(filepath).astype({ 'sample-id': int, 'origin-patient-id': str })

    def __load_loader_split(self) -> None:
        filepath = os.path.join(self.__path, 'loader-split.csv')
        if os.path.exists(filepath):
            self.__loader_split = pd.read_csv(filepath).astype({ 'sample-id': str })
        else:
            self.__loader_split = None

    def __str__(self) -> str:
        return self.__global_id
