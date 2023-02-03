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
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'nifti', name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

    @property
    def anon_index(self) -> Optional[pd.DataFrame]:
        filepath = os.path.join(self.__path, 'anon-index.csv')
        return pd.read_csv(filepath).astype({ 'anon-id': str, 'origin-patient-id': str })
    
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

    def list_regions(self) -> pd.DataFrame:
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        pat_ids = self.list_patients()

        # Add patient regions.
        for pat_id in pat_ids:
            for region in self.patient(pat_id).list_regions():
                data = {
                    'patient-id': pat_id,
                    'region': region,
                }
                df = append_row(df, data)

        # Set column types.
        df = df.astype(cols)

        return df

    def patient(
        self,
        id: Union[int, str]) -> NIFTIPatient:
        return NIFTIPatient(self, id)

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

    def __str__(self) -> str:
        return self.__global_id
    