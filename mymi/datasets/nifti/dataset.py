import numpy as np
import os
from pandas import DataFrame, read_csv
import re
from typing import *

from mymi import config
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ..dataset import CT_FROM_REGEXP, Dataset, DatasetType
from ..mixins import IndexMixin
from ..region_map import RegionMap
from .patient import NiftiPatient

class NiftiDataset(Dataset, IndexMixin):
    def __init__(
        self,
        id: str):
        self.__id = id
        self.__path = os.path.join(config.directories.datasets, 'nifti', self.__id)
        ct_from_id = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from_id = match.group(1)
        self.__ct_from = NiftiDataset(ct_from_id) if ct_from_id is not None else None
        self.__global_id = f"NIFTI:{self.__id}__CT_FROM_{self.__ct_from}__" if self.__ct_from is not None else f"NIFTI:{self.__id}"
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset '{self}' not found.")

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_index'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    def has_patients(
        self,
        pat_ids: PatientID = 'all',
        all: bool = False) -> bool:
        all_pats = self.list_patients()
        subset_pats = self.list_patients(pat_ids=pat_ids)
        n_overlap = len(np.intersect1d(all_pats, subset_pats))
        return n_overlap == len(all_pats) if all else n_overlap > 0

    # 'list_landmarks' can accept 'landmarks' keyword to filter - saves code elsewhere.
    def list_landmarks(self, *args, **kwargs) -> List[str]:
        # Load each patient.
        landmarks = []
        pat_ids = self.list_patients()
        for pat_id in pat_ids:
            pat_landmarks = self.patient(pat_id).list_landmarks(*args, **kwargs)
            landmarks += pat_landmarks
        landmarks = list(sorted(np.unique(landmarks)))
        return landmarks

    def list_patients(
        self,
        exclude: Optional[PatientIDs] = None,
        pat_ids: PatientIDs = 'all',    # Saves on filtering code elsewhere.
        region_ids: RegionIDs = 'all',
        splits: Splits = 'all') -> List[PatientID]:
        # Load patients from filenames.
        dirpath = os.path.join(self.__path, 'data', 'patients')
        all_ids = list(sorted(os.listdir(dirpath)))

        # Filter by 'region_ids'.
        ids = all_ids.copy()
        if region_ids != 'all':
            region_ids = regions_to_list(region_ids)
            all_region_ids = self.list_regions()

            # # Check that 'region_ids' are valid.
            # for r in region_ids:
            #     if not r in all_region_ids:
            #         logging.warning(f"Filtering by region '{r}' but it doesn't exist in dataset '{self}'.")

            def filter_fn(p: PatientID) -> bool:
                pat_region_ids = self.patient(p).list_regions(region_ids=region_ids)
                if len(pat_region_ids) > 0:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'splits'.
        if splits != 'all':
            filepath = os.path.join(self.__path, 'splits.csv')
            if not os.path.exists(filepath):
                raise ValueError(f"Holdout split file doesn't exist for dataset '{self}'.")
            split_df = load_csv(filepath)
            splits = arg_to_list(splits, Split, literals={ 'all': self.list_splits })
            all_splits = self.list_splits()

            # # Check that 'region_ids' are valid.
            # for s in splits:
            #     if not s in all_splits:
            #         logging.warning(f"Filtering by split '{s}' but it doesn't exist in dataset '{self}'.")

            def filter_fn(pat_id: PatientID) -> bool:
                pat_split = split_df[split_df['patient-id'] == pat_id].iloc[0]['split']
                if pat_split in splits:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'exclude'.
        if exclude is not None:
            exclude = arg_to_list(exclude, PatientID)
            ids = [p for p in ids if p not in exclude]

        # Filter by 'pat_ids'.
        if pat_ids != 'all':
            pat_ids = arg_to_list(pat_ids, PatientID)

            # # Check that 'pat_ids' are valid.
            # for p in pat_ids:
            #     if not p in all_ids:
            #         logging.warning(f"Filtering by patient ID '{p}' but it doesn't exist in dataset '{self}'.")

            filt_ids = []
            for i, id in enumerate(ids):
                # Check if any of the passed 'pat_ids' references this ID.
                for j, pid in enumerate(pat_ids):
                    if pid.startswith('idx:'):
                        if '-' in pid and not 'idx:-' in pid:   # Make sure negative indexing doesn't match - probably better with a regexp.
                            # Format 'idx:4-8'.
                            min_idx, max_idx = pid.split(':')[1].split('-')
                            min_idx, max_idx = int(min_idx), int(max_idx)
                            if i >= min_idx and i < max_idx:
                                filt_ids.append(id)
                                break
                        else:
                            # Format: 'idx:4' or 'idx:-4'.
                            idx = int(pid.split(':')[1])
                            if i == idx or (idx < 0 and i == len(ids) + idx):   # Allow negative indexing.
                                filt_ids.append(id)
                                break
                    elif id == pid:
                        filt_ids.append(id)
                        break
            ids = filt_ids
                        
            # if isinstance(pat_ids, str):
            #     # Check for special group format.
            #     regexp = r'^group:(\d+):(\d+)$'
            #     match = re.match(regexp, pat_ids)
            #     if match:
            #         group = int(match.group(1))
            #         num_groups = int(match.group(2))
            #         group_size = int(np.ceil(len(pat_ids) / num_groups))
            #         pat_ids = pat_ids[group * group_size:(group + 1) * group_size]
            #     else:
            #         pat_ids = arg_to_list(pat_ids, PatientID)
            # else:
            #     pat_ids = arg_to_list(pat_ids, PatientID)

        return ids

    def list_regions(
        self,
        pat_ids: PatientIDs = 'all') -> List[RegionID]:
        # Load all patients.
        pat_ids = self.list_patients(pat_ids=pat_ids)

        # Get patient regions.
        ids = []
        for p in pat_ids:
            pat_region_ids = self.patient(p).list_regions()
            ids += pat_region_ids
        ids = list(str(i) for i in np.unique(ids))

        return ids

    def list_splits(self) -> List[Split]:
        filepath = os.path.join(self.__path, 'splits.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"Holdout split file doesn't exist for dataset '{self}'.")
        split_df = load_csv(filepath)
        splits = list(sorted(split_df['split'].unique()))
        return splits

    def __load_data(self) -> None:
        # Load index.
        filepath = os.path.join(self.path, 'index.csv')
        if os.path.exists(filepath):
            map_types = { 'dicom-patient-id': str, 'patient-id': str, 'study-id': str, 'data-id': str }
            self._index = load_csv(filepath, map_types=map_types)
        else:
            self._index = None
    
        # Load excluded labels.
        filepath = os.path.join(self.__path, 'excluded-labels.csv')
        if os.path.exists(filepath):
            self.__excluded_labels = read_csv(filepath).astype({ 'patient-id': str })
            self.__excluded_labels = self.__excluded_labels.sort_values(['patient-id', 'region'])

            # Drop duplicates.
            dup_cols = ['patient-id', 'region']
            dup_df = self.__excluded_labels[self.__excluded_labels[dup_cols].duplicated()]
            if len(dup_df) > 0:
                logging.warning(f"Found {len(dup_df)} duplicate entries in 'excluded-labels.csv', removing.")
                self.__excluded_labels = self.__excluded_labels[~self.__excluded_labels[dup_cols].duplicated()]
        else:
            self.__excluded_labels = None

        # Load group index.
        filepath = os.path.join(self.__path, 'group-index.csv')
        if os.path.exists(filepath):
            self.__group_index = read_csv(filepath).astype({ 'patient-id': str })
        else:
            self.__group_index = None

        # Load region map.
        filepath = os.path.join(self.path, 'region-map.yaml')
        if os.path.exists(filepath):
            self.__region_map = RegionMap(load_yaml(filepath))
        else:
            self.__region_map = None

    @property
    def n_patients(self) -> int:
        return len(self.list_patients())

    @ensure_loaded
    def patient(
        self,
        id: Optional[PatientID] = None,
        n: Optional[int] = None,
        **kwargs) -> NiftiPatient:
        id = handle_idx_prefix(id, self.list_patients)
        if n is not None:
            if id is not None:
                raise ValueError("Cannot specify both 'id' and 'n'.")
            id = self.list_patients()[n]

        # Filter indexes to include only rows relevant to the new patient.
        index = self._index[self._index['patient-id'] == str(id)].copy() if self._index is not None else None
        exc_df = self.__excluded_labels[self.__excluded_labels['patient-id'] == str(id)].copy() if self.__excluded_labels is not None else None

        # Check that patient ID exists.
        dirpath = os.path.join(self.__path, 'data', 'patients', str(id))
        if not os.path.exists(dirpath):
            raise ValueError(f"Patient '{id}' not found for dataset '{self}'. Filepath: {dirpath}")

        return NiftiPatient(self.__id, id, dirpath, ct_from=self.__ct_from, index=index, excluded_labels=exc_df, region_map=self.__region_map, **kwargs)

    # Copied from 'mymi/reports/dataset/nift.py' to avoid circular dependency.
    def __load_patient_regions_report(
        self,
        exists_only: bool = False) -> Union[DataFrame, bool]:
        filepath = os.path.join(self.__path, 'reports', 'region-count.csv')
        if os.path.exists(filepath):
            if exists_only:
                return True
            else:
                return read_csv(filepath)
        else:
            if exists_only:
                return False
            else:
                raise ValueError(f"Patient regions report doesn't exist for dataset '{self}'.")

    @property
    @ensure_loaded
    def region_map(self) -> Optional[RegionMap]:
        return self.__region_map

    def __str__(self) -> str:
        return self.__global_id

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI
    
# Add properties.
props = ['ct_from', 'global_id', 'id', 'path']
for p in props:
    setattr(NiftiDataset, p, property(lambda self, p=p: getattr(self, f'_{NiftiDataset.__name__}__{p}')))
