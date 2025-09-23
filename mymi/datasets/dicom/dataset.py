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
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ..dataset import CT_FROM_REGEXP, Dataset, DatasetType
from ..mixins import IndexWithErrorsMixin
from ..region_map import RegionMap
from .index import INDEX_COLS, ERROR_INDEX_COLS, build_index as build_index_base, exists as index_exists
from .patient import DicomPatient

Z_SPACING_ROUND_DP = 2

class DicomDataset(Dataset, IndexWithErrorsMixin):
    def __init__(
        self,
        id: DatasetID) -> None:
        self.__path = os.path.join(config.directories.datasets, 'dicom', str(id))
        ct_from = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from = match.group(1)
        ct_from = NiftiDataset(ct_from) if ct_from is not None else None
        super().__init__(id, ct_from=ct_from)

    def build_index(
        self,
        **kwargs) -> None:
        build_index_base(self._id, **kwargs)

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_index'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    def has_patient(
        self,
        pat_id: PatientIDs,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_patients(pat_id=pat_id, **kwargs)
        req_ids = arg_to_list(pat_id, PatientID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    @ensure_loaded
    def list_patients(
        self,
        pat_id: PatientIDs = 'all', 
        region_id: RegionIDs = 'all',
        show_progress: bool = False,
        use_mapping: bool = True,
        use_regions_report: bool = True) -> List[str]:
        if region_id != 'all' and use_regions_report:
            # Can't use 'load_patient_regions_report' due to circularity.
            filename = 'regions-count.csv' if use_mapping else 'unmapped-regions-count.csv'
            filepath = os.path.join(self.__path, 'data', 'reports', filename)
            if os.path.exists(filepath):
                region_ids = regions_to_list(region_id)
                df = pd.read_csv(filepath)
                df = df[df['region'].isin(region_ids)]
                ids = list(sorted(df['patient-id'].unique()))
                return ids
            else:
                logging.warning(f"No patient regions report for dataset '{self}'. Would speed up queries filtered by 'region_id'.")

        # Load patient IDs from index.
        ids = list(sorted(self._index['patient-id'].unique()))

        # Filter on 'region_id'.
        if region_id != 'all':
            def filter_fn(p: PatientID) -> bool:
                pat_region_ids = self.patient(p).list_regions(region_id=region_id, use_mapping=use_mapping)
                return True if len(pat_region_ids) > 0 else False
            ids = list(filter(filter_fn, tqdm(ids, disable=not show_progress)))

        # Filter on 'pat_id'.
        if pat_id != 'all':
            pat_ids = arg_to_list(pat_id, PatientID)
            all_ids = ids.copy()
            ids = []
            for i, id in enumerate(all_ids):
                # Check if any of the passed 'pat_ids' references this ID.
                for j, pid in enumerate(pat_ids):
                    if pid.startswith('idx:'):
                        # Check if idx refer
                        idx = int(pid.split(':')[1])
                        if i == idx or (idx < 0 and i == len(all_ids) + idx):   # Allow negative indexing.
                            ids.append(id)
                            break
                    elif id == pid:
                        ids.append(id)
                        break

        return ids

    @ensure_loaded
    def patient(
        self,
        id: PatientID,
        **kwargs: Dict) -> DicomPatient:
        id = handle_idx_prefix(id, self.list_patients)
        if not self.has_patients(id):
            raise ValueError(f"Patient '{id}' not found in dataset '{self}'.")
        index = self._index[self._index['patient-id'] == str(id)]
        index_errors = self._index_errors[self._index_errors['patient-id'] == str(id)]
        ct_from = self.__ct_from.patient(id) if self.__ct_from is not None and self.__ct_from.has_patients(id) else None
        return DicomPatient(self._id, id, index, self._index_policy, index_errors, ct_from=ct_from, region_map=self.__region_map, **kwargs)

    @ensure_loaded
    def list_regions(
        self,
        pat_id: PatientIDs = 'all',
        use_mapping: bool = True) -> List[RegionID]:
        # Load all patients.
        pat_ids = self.list_patients(pat_id=pat_id)

        # Get patient regions.
        ids = []
        for p in tqdm(pat_ids):
            pat_region_ids = self.patient(p).list_regions(use_mapping=use_mapping)
            ids += pat_region_ids
        ids = list(np.unique(ids))

        return ids

    def __load_data(self) -> None:
        # Trigger index build if necessary.
        if not index_exists(self._id):
            build_index_base(self._id) 

        # Load index policy.
        filepath = os.path.join(self.__path, 'index-policy.yaml')
        self._index_policy = load_yaml(filepath)

        # Load index.
        filepath = os.path.join(self.__path, 'index.csv')
        try:
            self._index = load_csv(filepath, map_types=INDEX_COLS, parse_cols='mod-spec')
        except pd.errors.EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self._index = pd.DataFrame(columns=INDEX_COLS.keys())

        # Load error index.
        try:
            filepath = os.path.join(self.__path, 'index-errors.csv')
            self._index_errors = load_csv(filepath, map_types=ERROR_INDEX_COLS, parse_cols='mod-spec')
        except pd.errors.EmptyDataError:
            logging.info(f"Error index empty for dataset '{self}'.")
            self._index_errors = pd.DataFrame(columns=ERROR_INDEX_COLS.keys())

        # Load region map.
        filepath = os.path.join(self.__path, 'region-map.yaml')
        if os.path.exists(filepath):
            self.__region_map = RegionMap(load_yaml(filepath))
        else:
            self.__region_map = None

    @property
    @ensure_loaded
    def region_map(self) -> RegionMap:
        return self.__region_map

    def __str__(self) -> str:
        if self.__ct_from is None:
            return f"DicomDataset({self._id})"
        else:
            return f"DicomDataset({self._id}, ct_from={self.__ct_from.id})"
