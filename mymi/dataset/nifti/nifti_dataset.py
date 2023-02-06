import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional, Union

from mymi import config
from mymi import types
from mymi.utils import append_row, load_csv

from ..dataset import Dataset, DatasetType
from .nifti_patient import NIFTIPatient

class NIFTIDataset(Dataset):
    def __init__(
        self,
        name: str):
        self.__global_id = f"NIFTI: {name}"
        self.__anon_index = None                # Lazy-loaded.
        self.__excluded_regions = None          # Lazy-loaded.
        self.__loaded_anon_index = False
        self.__loaded_excluded_regions = False
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'nifti', name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

    @property
    def anon_index(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_anon_index:
            self.__load_anon_index()
            self.__loaded_anon_index = True
        return self.__anon_index

    @property
    def excluded_regions(self) -> Optional[pd.DataFrame]:
        if not self.__loaded_excluded_regions:
            self.__load_excluded_regions()
            self.__loaded_excluded_regions = True
        return self.__excluded_regions
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def path(self) -> str:
        return self.__path

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    def list_patients(
        self,
        regions: types.PatientRegions = 'all') -> List[str]:
        # Load patients.
        ct_path = os.path.join(self.__path, 'data', 'ct')
        files = list(sorted(os.listdir(ct_path)))
        pats = [f.replace('.nii.gz', '') for f in files]

        # Filter by 'regions'.
        pats = list(filter(self.__filter_patient_by_regions(regions), pats))
        return pats

    def patient(
        self,
        id: Union[int, str]) -> NIFTIPatient:
        return NIFTIPatient(self, id, excluded_regions=self.excluded_regions)

    def __filter_patient_by_regions(
        self,
        regions: types.PatientRegions) -> Callable[[str], bool]:
        def func(id):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.patient(id).has_region(regions)
            else:
                pat_regions = self.patient(id).list_regions()
                if len(np.intersect1d(regions, pat_regions)) != 0:
                    return True
                else:
                    return False
        return func
    
    def __load_anon_index(self) -> None:
        filepath = os.path.join(self.__path, 'anon-index.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"No 'anon-index.csv' found for '{self}'.")
        self.__anon_index = pd.read_csv(filepath).astype({ 'anon-id': str, 'origin-patient-id': str })
    
    def __load_excluded_regions(self) -> None:
        filepath = os.path.join(self.__path, 'excluded-regions.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"No 'excluded-regions.csv' found for '{self}'.")
        self.__excluded_regions = pd.read_csv(filepath).astype({ 'patient-id': str })

    def __str__(self) -> str:
        return self.__global_id
    