import numpy as np
import os
import pandas as pd
from typing import Callable, List, Optional

from mymi import config
from mymi import types

from ..dataset import Dataset, DatasetType
from .nifti_patient import NIFTIPatient

class NIFTIDataset(Dataset):
    def __init__(
        self,
        name: str):
        self._global_id = f"NIFTI: {name}"
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'nifti', name)
        if not os.path.exists(self._path):
            raise ValueError(f"Dataset '{self}' not found.")
    
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
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    @property
    def anon_manifest(self) -> Optional[pd.DataFrame]:
        man_df = config.load_csv('anon-maps', f'{self._name}.csv')
        return man_df

    def list_patients(
        self,
        regions: types.PatientRegions = 'all') -> List[str]:
        """
        returns: a list of NIFTI IDs.
        """
        # Load patients.
        ct_path = os.path.join(self._path, 'data', 'ct')
        files = list(sorted(os.listdir(ct_path)))
        pats = [f.replace('.nii.gz', '') for f in files]

        # Filter by 'regions'.
        pats = list(filter(self._filter_patient_by_regions(regions), pats))
        return pats

    def list_regions(self) -> pd.DataFrame:
        """
        returns: a DataFrame with patient region names.
        """
        # Define table structure.
        cols = {
            'patient-id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        ids = self.list_patients()

        # Add patient regions.
        for id in ids:
            for region in self.patient(id).list_regions():
                data = {
                    'patient-id': id,
                    'region': region,
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    def patient(
        self,
        id: str) -> NIFTIPatient:
        return NIFTIPatient(self, id)

    def _filter_patient_by_regions(
        self,
        regions: types.PatientRegions) -> Callable[[str], bool]:
        """
        returns: a function that filters patients on region presence.
        args:
            regions: the passed 'regions' kwarg.
        """
        def fn(id):
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
        return fn
