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
        self._path = os.path.join(config.directories.datasets, 'dicom', str(id))
        if not os.path.exists(self._path):
            raise ValueError(f"No dicom dataset '{id}' found at path: {self._path}")
        ct_from = None
        for f in os.listdir(self._path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from = match.group(1)
        ct_from = NiftiDataset(ct_from) if ct_from is not None else None
        super().__init__(id, ct_from=ct_from)

    def build_index(
        self,
        **kwargs) -> None:
        build_index_base(self._id, **kwargs)

    def has_patient(
        self,
        pat: PatientIDs,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_patients(pat=pat, **kwargs)
        req_ids = arg_to_list(pat, PatientID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    @Dataset.ensure_loaded
    def list_patients(
        self,
        group: PatientGroups = 'all',
        pat: PatientIDs = 'all', 
        region: RegionIDs = 'all',
        show_progress: bool = False,
        use_mapping: bool = True,
        use_regions_report: bool = True) -> List[str]:
        if region != 'all' and use_regions_report:
            # Can't use 'load_patient_regions_report' due to circularity.
            filename = 'regions.csv' if use_mapping else 'unmapped-regions.csv'
            filepath = os.path.join(self._path, 'reports', filename)
            if os.path.exists(filepath):
                regions = regions_to_list(region)
                df = pd.read_csv(filepath)
                df = df[df['region'].isin(regions)]
                ids = list(sorted(df['patient-id'].unique()))
            else:
                logging.warning(f"No patient regions report for dataset '{self}'. Would speed up queries filtered by 'region'.")
        else:
            # Load patient IDs from index.
            ids = list(sorted(self._index['patient-id'].unique()))

            # Filter by 'region'.
            if region != 'all':
                def filter_fn(p: PatientID) -> bool:
                    pat_regions = self.patient(p).list_regions(region=region, use_mapping=use_mapping)
                    return True if len(pat_regions) > 0 else False
                ids = list(filter(filter_fn, tqdm(ids, disable=not show_progress)))

        # Filter by 'pat'.
        if pat != 'all':
            pat_ids = arg_to_list(pat, PatientID)
            all_ids = ids.copy()
            ids = []
            for i, id in enumerate(all_ids):
                # Check if any of the passed 'pat_ids' references this ID.
                for j, pid in enumerate(pat_ids):
                    if pid.startswith('i:'):
                        # Check if idx refer
                        idx = int(pid.split(':')[1])
                        if i == idx or (idx < 0 and i == len(all_ids) + idx):   # Allow negative indexing.
                            ids.append(id)
                            break
                    elif id == pid:
                        ids.append(id)
                        break

        # Filter by 'group'.
        if group != 'all':
            if self._groups is None:
                raise ValueError(f"File 'groups.csv' not found for dicom dataset '{self._id}'.")
            all_groups = self.list_groups()
            groups = arg_to_list(group, PatientGroup, literals={ 'all': all_groups })
            for g in groups:
                if g not in all_groups:
                    raise ValueError(f"Group '{g}' not found.")

            def filter_fn(p: PatientID) -> bool:
                pat_groups = self._groups[self._groups['patient-id'] == p]
                if len(pat_groups) == 0:
                    return False
                elif len(pat_groups) > 1:
                    raise ValueError(f"Patient '{p}' is a member of more than one group.")
                pat_group = pat_groups.iloc[0]['group-id']
                if pat_group in groups:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        return ids

    @Dataset.ensure_loaded
    def patient(
        self,
        id: PatientID,
        group: PatientGroups = 'all',
        **kwargs: Dict,
        ) -> DicomPatient:
        id = handle_idx_prefix(id, lambda: self.list_patients(group=group))
        if not self.has_patient(id):
            raise ValueError(f"Patient '{id}' not found in dataset '{self}'.")
        index = self._index[self._index['patient-id'] == str(id)]
        index_errors = self._index_errors[self._index_errors['patient-id'] == str(id)]
        ct_from = self._ct_from.patient(id) if self._ct_from is not None and self._ct_from.has_patient(id) else None
        return DicomPatient(self, id, index, self._index_policy, index_errors, config=self._config, ct_from=ct_from, region_map=self.__region_map, **kwargs)

    @Dataset.ensure_loaded
    def list_regions(
        self,
        group: PatientGroups = 'all',
        pat: PatientIDs = 'all',
        use_mapping: bool = True,
        use_regions_report: bool = True,
        ) -> List[RegionID]:
        # Load all patients.
        pat_ids = self.list_patients(group=group, pat=pat)

        if use_regions_report:
            # Can't use 'load_patient_regions_report' due to circularity.
            filename = 'regions.csv' if use_mapping else 'unmapped-regions.csv'
            filepath = os.path.join(self._path, 'reports', filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df = df[df['patient-id'].isin(pat_ids)]
                ids = list(sorted(df['region'].unique()))
                return ids
            else:
                logging.warning(f"No regions report for dataset '{self}'. Would speed up 'list_regions' query.")

        # Get patient regions.
        # Trawl the depths for region IDs.
        ids = []
        for p in tqdm(pat_ids):
            pat = self.patient(p)
            study_ids = pat.list_studies()
            for s in study_ids:
                study = pat.study(s)
                series_ids = study.list_rtstruct_series()
                for s in series_ids:
                    series = study.rtstruct_series(s)
                    ids += series.list_regions(use_mapping=use_mapping)
        ids = list(np.unique(ids))

        return ids

    def _load_data(self) -> None:
        # Trigger index build if necessary.
        if not index_exists(self._id):
            build_index_base(self._id) 

        # Load groups.
        filepath = os.path.join(self._path, 'groups.csv')
        self._groups = load_csv(filepath) if os.path.exists(filepath) else None

        # Load index policy.
        filepath = os.path.join(self._path, 'index-policy.yaml')
        self._index_policy = load_yaml(filepath)

        # Load index.
        filepath = os.path.join(self._path, 'index.csv')
        try:
            self._index = load_csv(filepath, map_types=INDEX_COLS, parse_cols='mod-spec')
        except pd.errors.EmptyDataError:
            logging.info(f"Index empty for dataset '{self}'.")
            self._index = pd.DataFrame(columns=INDEX_COLS.keys())

        # Load error index.
        try:
            filepath = os.path.join(self._path, 'index-errors.csv')
            self._index_errors = load_csv(filepath, map_types=ERROR_INDEX_COLS, parse_cols='mod-spec')
        except pd.errors.EmptyDataError:
            logging.info(f"Error index empty for dataset '{self}'.")
            self._index_errors = pd.DataFrame(columns=ERROR_INDEX_COLS.keys())

        # Load region map.
        filepath = os.path.join(self._path, 'region-map.yaml')
        if os.path.exists(filepath):
            self.__region_map = RegionMap(load_yaml(filepath))
        else:
            self.__region_map = None

    @property
    @Dataset.ensure_loaded
    def region_map(self) -> RegionMap:
        return self.__region_map

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
