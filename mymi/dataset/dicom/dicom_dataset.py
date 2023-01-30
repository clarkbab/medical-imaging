import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Union

from mymi import config
from mymi import logging
from mymi import regions
from mymi import types

from ..dataset import Dataset, DatasetType
from .dicom_patient import DICOMPatient
from .index import ERRORS_COLS, INDEX_COLS, build_index
from .region_map import RegionMap

Z_SPACING_ROUND_DP = 2

class DICOMDataset(Dataset):
    def __init__(
        self,
        name: str):
        self.__path = os.path.join(config.directories.datasets, 'dicom', name)
        self.__index = None             # Lazy-loaded.
        self.__index_errors = None      # Lazy-loaded.
        self.__region_map = None        # Lazy-loaded.

        # Load 'ct_from' flag.
        ct_from_name = None
        for f in os.listdir(self.__path):
            match = re.match('^ct_from_(.*)$', f)
            if match:
                ct_from_name = match.group(1)

        self.__ct_from = DICOMDataset(ct_from_name) if ct_from_name is not None else None
        self.__global_id = f"DICOM: {name}"
        self.__global_id = self.__global_id + f" (CT from - {self.__ct_from})" if self.__ct_from is not None else self.__global_id
        self.__name = name
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

    @property
    def ct_from(self) -> Optional['DICOMDataset']:
        return self.__ct_from

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def index(self) -> pd.DataFrame:
        if self.__index is None:
            self.__load_index()
        return self.__index

    @property
    def index_errors(self) -> pd.DataFrame:
        if self.__index_errors is None:
            self.__load_index_errors()
        return self.__index_errors

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    @property
    def region_map(self) -> RegionMap:
        if self.__region_map is None:
            self.__load_region_map()
        return self.__region_map

    @property
    def type(self) -> DatasetType:
        return self._type

    def has_patient(
        self,
        id: types.PatientID) -> bool:
        return id in self.list_patients()

    def list_patients(
        self,
        regions: types.PatientRegions = 'all') -> List[str]:
        pats = list(sorted(self.index['patient-id'].unique()))

        # Filter by 'regions'.
        pats = list(filter(self.__filter_patient_by_regions(regions), pats))
        return pats

    def patient(
        self,
        id: types.PatientID,
        **kwargs: Dict) -> DICOMPatient:
        return DICOMPatient(self, id, region_map=self.region_map, **kwargs)

    def list_regions(
        self,
        pat_ids: types.PatientIDs = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        # Filter patients.
        pats = self.list_patients()
        pats = list(filter(self.__filter_patient_by_pat_ids(pat_ids), pats))

        # Get patient regions.
        regions = []
        for pat in tqdm(pats):
            pat_regions = self.patient(pat).list_regions(use_mapping=use_mapping)
            regions += pat_regions

        return list(np.unique(regions))

    def rebuild_index(self) -> None:
        build_index(self.__name)

    def __filter_patient_by_pat_ids(
        self,
        pat_ids: Union[str, List[str]]) -> Callable[[str], bool]:
        def fn(id):
            if ((isinstance(pat_ids, str) and (pat_ids == 'all' or id == pat_ids)) or
                ((isinstance(pat_ids, list) or isinstance(pat_ids, np.ndarray) or isinstance(pat_ids, tuple)) and id in pat_ids)):
                return True
            else:
                return False
        return fn

    def __filter_patient_by_regions(
        self,
        regions: types.PatientRegions,
        use_mapping: bool = True) -> Callable[[str], bool]:
        def fn(id):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.patient(id).has_region(regions, use_mapping=use_mapping)
            else:
                pat_regions = self.patient(id).list_regions(use_mapping=use_mapping)
                if len(np.intersect1d(regions, pat_regions)) != 0:
                    return True
                else:
                    return False
        return fn

    def __load_index(self) -> None:
        filepath = os.path.join(self.__path, 'index.csv')
        if not os.path.exists(filepath):
            build_index(self.__name)
        try:
            self.__index = pd.read_csv(filepath, dtype={ 'patient-id': str })
        except pd.errors.EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self.__index = pd.DataFrame(columns=INDEX_COLS.keys())

    def __load_index_errors(self) -> None:
        filepath = os.path.join(self.__path, 'index-errors.csv')
        if not os.path.exists(filepath):
            build_index(self.__name)
        try:
            self.__index_errors = pd.read_csv(filepath, dtype={ 'patient-id': str })
        except pd.errors.EmptyDataError:
            logging.info(f"Index-errors empty for dataset '{self}'.")
            self.__index_errors = pd.DataFrame(columns=ERRORS_COLS.keys())

    def __load_region_map(self) -> Optional[RegionMap]:
        filepath = os.path.join(self.__path, 'region-map.csv')
        self.__region_map = RegionMap.load(filepath)

    def __str__(self) -> str:
        return self.__global_id
