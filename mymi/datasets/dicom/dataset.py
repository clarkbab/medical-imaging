import ast
import json
import numpy as np
import os
import pandas as pd
import re
from tqdm import tqdm
from typing import *

from mymi import config
from mymi import logging
from mymi.typing import *
from mymi.utils import *

from ..dataset import Dataset, DatasetType
from ..shared import CT_FROM_REGEXP
from .patient import DicomPatient
from .index import INDEX_COLS, ERROR_INDEX_COLS, INDEX_INDEX_COL, build_index, exists as index_exists
from .files.rtstruct import RegionMap, DEFAULT_POLICY as DEFAULT_REGION_POLICY

Z_SPACING_ROUND_DP = 2

class DicomDataset(Dataset):
    def __init__(
        self,
        id: str,
        rebuild_index: bool = False) -> None:
        # Create 'global ID'.
        self.__id = id
        self.__path = os.path.join(config.directories.datasets, 'dicom', self.__id)
        ct_from_id = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from_id = match.group(1)
        self.__ct_from = DicomDataset(ct_from_id) if ct_from_id is not None else None
        self.__global_id = f"DICOM:{self.__id}__CT_FROM_{self.__ct_from}__" if self.__ct_from is not None else f"DICOM:{self.__id}"
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found. Filepath: {self.__path}.")

        if rebuild_index:
            build_index(self.__id)

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__index'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    @ensure_loaded
    def error_index(self) -> pd.DataFrame:
        return self.__error_index

    @property
    @ensure_loaded
    def index_policy(self) -> Dict[str, Any]:
        return self.__index_policy

    @property
    @ensure_loaded
    def region_policy(self) -> Dict[str, Any]:
        return self.__region_policy

    @property
    @ensure_loaded
    def region_map(self) -> RegionMap:
        return self.__region_map

    def rebuild_index(self) -> None:
        build_index(self.__id)

    def has_patient(
        self,
        id: PatientID) -> bool:
        return id in self.list_patients()

    @ensure_loaded
    def list_patients(
        self,
        pat_ids: PatientIDs = 'all', 
        region_ids: RegionIDs = 'all',
        show_progress: bool = False,
        use_mapping: bool = True,
        use_patient_regions_report: bool = True) -> List[str]:

        # Use patient regions report to accelerate 'list_patients' if filtering on regions.
        if region_ids != 'all' and use_patient_regions_report and self.__load_patient_regions_report(exists_only=True, use_mapping=use_mapping):
            logging.info(f"Using patient regions report to accelerate 'list_patients' (filtered by region).")
            df = self.__load_patient_regions_report(use_mapping=use_mapping)
            df = df[df['region'].isin(region_ids)]
            ids = list(sorted(df['patient-id'].unique()))
            return ids

        # Load patient IDs from index.
        ids = list(sorted(self.index['patient-id'].unique()))

        # Filter on 'region_ids'.
        if region_ids != 'all':
            def filter_fn(p: PatientID) -> bool:
                pat_region_ids = self.patient(p).list_regions(region_ids=region_ids, use_mapping=use_mapping)
                if len(pat_region_ids) > 0:
                    return True
                else:
                    return False
            if show_progress:
                ids = tqdm(ids)
            ids = list(filter(filter_fn, ids))

        # Filter on 'pat_ids'.
        if pat_ids != 'all':
            ids = [i for i in ids if i in arg_to_list(pat_ids, PatientID)]

        return ids

    @ensure_loaded
    def patient(
        self,
        id: PatientID,
        **kwargs: Dict) -> DicomPatient:
        return DicomPatient(self, id, region_map=self.region_map, **kwargs)

    @ensure_loaded
    def list_regions(
        self,
        pat_ids: PatientIDs = 'all',
        use_mapping: bool = True) -> pd.DataFrame:
        # Load all patients.
        pat_ids = self.list_patients(pat_ids=pat_ids)

        # Get patient regions.
        ids = []
        for p in tqdm(pat_ids):
            pat_region_ids = self.patient(p).list_regions(use_mapping=use_mapping)
            ids += pat_region_ids
        ids = list(np.unique(ids))

        return ids

    def __load_data(self) -> None:
        # Trigger index build if necessary.
        if not index_exists(self.__id):
            build_index(self.__id) 

        # Load index policy.
        filepath = os.path.join(self.__path, 'index-policy.yaml')
        self.__index_policy = load_yaml(filepath)

        # Load index.
        filepath = os.path.join(self.__path, 'index.csv')
        try:
            self.__index = pd.read_csv(filepath, dtype={ 'patient-id': str }, index_col=INDEX_INDEX_COL)
            self.__index['mod-spec'] = self.__index['mod-spec'].apply(lambda m: ast.literal_eval(m))      # Convert str to dict.
        except pd.errors.EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self.__index = pd.DataFrame(columns=INDEX_COLS.keys())

        # Load index errors.
        try:
            filepath = os.path.join(self.__path, 'index-errors.csv')
            self.__error_index = pd.read_csv(filepath, dtype={ 'patient-id': str }, index_col=INDEX_INDEX_COL)
        except pd.errors.EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self.__error_index = pd.DataFrame(columns=ERROR_INDEX_COLS.keys())

        # Load region map.
        filepath = os.path.join(self.path, 'region-map.csv')
        if os.path.exists(filepath):
            self.__region_map = RegionMap.load(filepath)
        else:
            self.__region_map = None
    
        # Load region policy.
        filepath = os.path.join(self.__path, 'region-policy.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.__region_policy = json.load(f)
        else:
            self.__region_policy = DEFAULT_REGION_POLICY

    # Copied from 'mymi/reporting/dataset/dicom.py' to avoid circular dependency.
    def __load_patient_regions_report(
        self,
        exists_only: bool = False,
        use_mapping: bool = True) -> Union[pd.DataFrame, bool]:
        filename = 'region-count.csv' if use_mapping else 'region-count-unmapped.csv'
        filepath = os.path.join(self.__path, 'reports', filename)
        if os.path.exists(filepath):
            if exists_only:
                return True
            else:
                return pd.read_csv(filepath)
        else:
            if exists_only:
                return False
            else:
                raise ValueError(f"Patient regions report doesn't exist for dataset '{self}'.")
    
# Add properties.
props = ['ct_from', 'global_id', 'id', 'path']
for p in props:
    setattr(DicomDataset, p, property(lambda self, p=p: getattr(self, f'_{DicomDataset.__name__}__{p}')))
