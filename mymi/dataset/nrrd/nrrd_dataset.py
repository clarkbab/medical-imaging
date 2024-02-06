import numpy as np
import os
import pandas as pd
from typing import List, Literal, Optional, Union

from mymi import config
from mymi import logging
from mymi.types import PatientRegions

from ..dataset import Dataset, DatasetType
from .nrrd_patient import NRRDPatient

class NRRDDataset(Dataset):
    def __init__(
        self,
        name: str):
        self.__global_id = f"NRRD: {name}"
        self.__index = None                # Lazy-loaded.
        self.__excluded_labels = None          # Lazy-loaded.
        self.__group_index = None               # Lazy-loaded.
        self.__loaded_index = False
        self.__loaded_excluded_labels = False
        self.__loaded_group_index = False
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'nrrd', name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

    @property
    def index(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_index:
            self.__load_index()
            self.__loaded_index = True
        return self.__index
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def excluded_labels(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_excluded_labels:
            self.__load_excluded_labels()
            self.__loaded_excluded_labels = True
        return self.__excluded_labels

    @property
    def group_index(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_group_index:
            self.__load_group_index()
            self.__loaded_group_index = True
        return self.__group_index

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def path(self) -> str:
        return self.__path

    @property
    def type(self) -> DatasetType:
        return DatasetType.NRRD

    def list_patients(
        self,
        labels: Literal['included', 'excluded', 'all'] = 'included',
        region: Optional[PatientRegions] = None) -> List[str]:

        # Load patients.
        ct_path = os.path.join(self.__path, 'data', 'ct')
        files = list(sorted(os.listdir(ct_path)))
        pat_ids = [f.replace('.nrrd', '') for f in files]

        # Filter by 'region'.
        if region is not None:
            pat_ids = list(filter(lambda pat_id: self.patient(pat_id).has_region(region, labels=labels), pat_ids))
        return pat_ids

    def list_regions(self) -> List[str]:
        # Load each patient.
        regions = []
        pat_ids = self.list_patients()
        for pat_id in pat_ids:
            pat_regions = self.patient(pat_id).list_regions()
            regions += pat_regions
        regions = list(sorted(np.unique(regions)))

        return regions

    def patient(
        self,
        id: Union[int, str]) -> NRRDPatient:
        return NRRDPatient(self, id, excluded_labels=self.excluded_labels)
    
    def __load_index(self) -> None:
        filepath = os.path.join(self.__path, 'index.csv.csv')
        if os.path.exists(filepath):
            self.__index = pd.read_csv(filepath).astype({ 'anon-id': str, 'origin-patient-id': str })
        else:
            self.__index = None
    
    def __load_excluded_labels(self) -> None:
        filepath = os.path.join(self.__path, 'excluded-labels.csv')
        if os.path.exists(filepath):
            self.__excluded_labels = pd.read_csv(filepath).astype({ 'patient-id': str })
            self.__excluded_labels = self.__excluded_labels.sort_values(['patient-id', 'region'])

            # Drop duplicates.
            dup_cols = ['patient-id', 'region']
            dup_df = self.__excluded_labels[self.__excluded_labels[dup_cols].duplicated()]
            if len(dup_df) > 0:
                logging.warning(f"Found {len(dup_df)} duplicate entries in 'excluded-labels.csv', removing.")
                self.__excluded_labels = self.__excluded_labels[~self.__excluded_labels[dup_cols].duplicated()]
        else:
            self.__excluded_labels = None

    def __load_group_index(self) -> None:
        filepath = os.path.join(self.__path, 'group-index.csv')
        if os.path.exists(filepath):
            self.__group_index = pd.read_csv(filepath).astype({ 'patient-id': str })
        else:
            self.__group_index = None

    def __str__(self) -> str:
        return self.__global_id
    