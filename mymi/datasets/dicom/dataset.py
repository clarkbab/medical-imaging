from ast import literal_eval
import json
import numpy as np
import os
from pandas import DataFrame, read_csv
from pandas.errors import EmptyDataError
import re
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Union

from mymi import config
from mymi import logging
from mymi.typing import PatientID, PatientIDs, PatientRegions
from mymi.utils import arg_to_list

from ..dataset import Dataset, DatasetType
from ..shared import CT_FROM_REGEXP
from .patient import DicomPatient
from .index import DEFAULT_POLICY as DEFAULT_INDEX_POLICY, INDEX_COLS, ERROR_INDEX_COLS, INDEX_INDEX_COL, build_index as build_index_fn
from .files.rtstruct import RegionMap, DEFAULT_POLICY as DEFAULT_REGION_POLICY

Z_SPACING_ROUND_DP = 2

class DicomDataset(Dataset):
    def __init__(
        self,
        name: str,
        build_index: bool = False) -> None:
        # Create 'global ID'.
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'dicom', self.__name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset 'DICOM: {self.__name}' not found. Filepath: {self.__path}.")
        ct_from_name = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from_name = match.group(1)
        self.__ct_from = DicomDataset(ct_from_name) if ct_from_name is not None else None
        self.__global_id = f"DICOM:{self.__name}__CT_FROM_{self.__ct_from}__" if self.__ct_from is not None else f"DICOM:{self.__name}"

        self.__index = None             # Lazy-loaded.
        self.__error_index = None      # Lazy-loaded.
        self.__index_policy = None      # Lazy-loaded.
        self.__loaded_region_dups = False
        self.__loaded_region_map = False
        self.__region_dups = None       # Lazy-loaded.
        self.__region_map = None        # Lazy-loaded.
        self.__region_policy = None     # Lazy-loaded.

        if build_index:
            self.build_index()

    @property
    def ct_from(self) -> Optional['DicomDataset']:
        return self.__ct_from

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def index(self) -> DataFrame:
        if self.__index is None:
            self.__load_index()
        return self.__index

    @property
    def error_index(self) -> DataFrame:
        if self.__error_index is None:
            self.__load_index()
        return self.__error_index

    @property
    def index_policy(self) -> Dict[str, Any]:
        if self.__index_policy is None:
            self.__load_index()
        return self.__index_policy

    @property
    def region_policy(self) -> Dict[str, Any]:
        if self.__region_policy is None:
            self.__load_region_policy()
        return self.__region_policy

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    @property
    def region_dups(self) -> DataFrame:
        if not self.__loaded_region_dups:
            self.__load_region_dups()
            self.__loaded_region_dups = True
        return self.__region_dups

    @property
    def region_map(self) -> RegionMap:
        if not self.__loaded_region_map:
            self.__load_region_map()
            self.__loaded_region_map = True
        return self.__region_map

    @property
    def type(self) -> DatasetType:
        return self._type

    def build_index(self) -> None:
        build_index_fn(self.__name)

    def has_patient(
        self,
        id: PatientID) -> bool:
        return id in self.list_patients()

    def list_patients(
        self,
        regions: Optional[PatientRegions] = None,
        show_progress: bool = False,
        use_mapping: bool = True,
        use_patient_regions_report: bool = True) -> List[str]:
        regions = arg_to_list(regions, str)

        # Use patient regions report to accelerate 'list_patients' if filtering on regions.
        if regions is not None and use_patient_regions_report and self.__load_patient_regions_report(exists_only=True, use_mapping=use_mapping):
            logging.info(f"Using patient regions report to accelerate 'list_patients' (filtered by region).")
            df = self.__load_patient_regions_report(use_mapping=use_mapping)
            df = df[df['region'].isin(regions)]
            pat_ids = list(sorted(df['patient-id'].unique()))
            return pat_ids

        # Load patient IDs from index.
        pat_ids = list(sorted(self.index['patient-id'].unique()))

        # Filter on 'regions'.
        if regions is not None:
            def filter_fn(pat_id):
                pat_regions = self.patient(pat_id).list_regions(regions=regions, use_mapping=use_mapping)
                if len(pat_regions) > 0:
                    return True
                else:
                    return False
            if show_progress:
                pat_ids = tqdm(pat_ids)
            pat_ids = list(filter(filter_fn, pat_ids))

        return pat_ids

    def patient(
        self,
        id: PatientID,
        **kwargs: Dict) -> DicomPatient:
        return DicomPatient(self, id, region_dups=self.region_dups, region_map=self.region_map, **kwargs)

    def list_regions(
        self,
        pat_id: Optional[PatientIDs] = None,
        use_mapping: bool = True) -> DataFrame:
        # Load all patients.
        pats = self.list_patients()

        # Filter on 'pat_ids'.
        if pat_id is not None:
            pat_ids = arg_to_list(pat_id, PatientID)
            def filter_fn(pat_id):
                if ((isinstance(pat_ids, str) and (pat_ids == 'all' or id == pat_ids)) or
                    ((isinstance(pat_ids, list) or isinstance(pat_ids, np.ndarray) or isinstance(pat_ids, tuple)) and id in pat_ids)):
                    return True
                else:
                    return False
            pats = list(filter(self.__filter_patient_by_pat_ids(pat_ids), pats))

        # Get patient regions.
        regions = []
        for pat in tqdm(pats):
            pat_regions = self.patient(pat).list_regions(use_mapping=use_mapping)
            regions += pat_regions

        return list(np.unique(regions))

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

    def __filter_patient_by_region(
        self,
        region: PatientRegions,
        use_mapping: bool = True) -> Callable[[str], bool]:
        regions = arg_to_list(region, str)

        def fn(pat_id):
            pat_regions = self.patient(pat_id).list_regions(regions=regions, use_mapping=use_mapping)
            if len(pat_regions) > 0:
                return True
            else:
                return False
        return fn

    def __load_index(self) -> None:
        # Load index policy.
        filepath = os.path.join(self.__path, 'index-policy.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.__index_policy = json.load(f)
        else:
            self.__index_policy = DEFAULT_INDEX_POLICY

        # Trigger index build if not present.
        filepath = os.path.join(self.__path, 'index.csv')
        if not os.path.exists(filepath):
            self.build_index()

        # Load index.
        try:
            self.__index = read_csv(filepath, dtype={ 'patient-id': str }, index_col=INDEX_INDEX_COL)
            self.__index['mod-spec'] = self.__index['mod-spec'].apply(lambda m: literal_eval(m))      # Convert str to dict.
        except EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self.__index = DataFrame(columns=INDEX_COLS.keys())

        # Load index errors.
        try:
            filepath = os.path.join(self.__path, 'index-errors.csv')
            self.__error_index = read_csv(filepath, dtype={ 'patient-id': str }, index_col=INDEX_INDEX_COL)
        except EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self.__index = DataFrame(columns=ERROR_INDEX_COLS.keys())

    # Copied from 'mymi/reporting/dataset/dicom.py' to avoid circular dependency.
    def __load_patient_regions_report(
        self,
        exists_only: bool = False,
        use_mapping: bool = True) -> Union[DataFrame, bool]:
        filename = 'region-count.csv' if use_mapping else 'region-count-unmapped.csv'
        filepath = os.path.join(self.__path, 'reports', filename)
        if os.path.exists(filepath):
            if exists_only:
                return True
            else:
                return read_csv(filepath)
        else:
            if exists_only:
                return False
            else:
                raise ValueError(f"Patient regions report doesn't exist for dataset '{dataset}'.")

    def __load_region_dups(self) -> Optional[DataFrame]:
        filepath = os.path.join(self.__path, 'region-dups.csv')
        if os.path.exists(filepath):
            self.__region_dups = read_csv(filepath)
        else:
            self.__region_dups = None

    def __load_region_map(self) -> Optional[RegionMap]:
        filepath = os.path.join(self.__path, 'region-map.csv')
        if os.path.exists(filepath):
            self.__region_map = RegionMap.load(filepath)
        else:
            self.__region_map = None
    
    def __load_region_policy(self) -> Dict[str, Any]:
        filepath = os.path.join(self.__path, 'region-policy.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.__region_policy = json.load(f)
        else:
            self.__region_policy = DEFAULT_REGION_POLICY

    def __str__(self) -> str:
        return self.__global_id
