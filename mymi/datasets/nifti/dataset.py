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

from ..dataset import Dataset, DatasetType
from ..shared import CT_FROM_REGEXP
from .patient import NiftiPatient

class NiftiDataset(Dataset):
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

        self.__index = None                     # Lazy-loaded.
        self.__excluded_labels = None           # Lazy-loaded.
        self.__group_index = None               # Lazy-loaded.
        self.__processed_labels = None          # Lazy-loaded.
        self.__region_map = None                # Lazy-loaded.
        self.__loaded_index = False
        self.__loaded_excluded_labels = False
        self.__loaded_group_index = False
        self.__loaded_processed_labels = False
        self.__loaded_region_map = False

    @property
    def index(self) -> Optional[DataFrame]:
        if not self.__loaded_index:
            self.__load_index()
            self.__loaded_index = True
        return self.__index

    @property
    def excluded_labels(self) -> Optional[DataFrame]:
        if not self.__loaded_excluded_labels:
            self.__load_excluded_labels()
            self.__loaded_excluded_labels = True
        return self.__excluded_labels

    @property
    def group_index(self) -> Optional[DataFrame]:
        if not self.__loaded_group_index:
            self.__load_group_index()
            self.__loaded_group_index = True
        return self.__group_index

    @property
    def n_patients(self) -> int:
        return len(self.list_patients())

    @property
    def processed_labels(self) -> Optional[DataFrame]:
        if not self.__loaded_processed_labels:
            self.__load_processed_labels()
            self.__loaded_processed_labels = True
        return self.__processed_labels

    @property
    def region_map(self) -> Optional[Dict[str, str]]:
        if not self.__loaded_region_map:
            self.__load_region_map()
            self.__loaded_region_map = True
        return self.__region_map

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    def has_patients(
        self,
        pat_ids: PatientID = 'all',
        all: bool = False) -> bool:
        all_pats = self.list_patients()
        subset_pats = self.list_patients(ids=pat_ids)
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

        if self.__ct_from is None:
            # Load patients from filenames.
            pat_path = os.path.join(self.__path, 'data', 'patients')
            ids = list(sorted(os.listdir(pat_path)))
        else:
            # Load patients from 'ct_from' dataset.
            ids = self.__ct_from.list_patients(region_ids=None)

        # Filter by 'region_ids'.
        if region_ids != 'all':
            region_ids = regions_to_list(region_ids)
            def filter_fn(p: PatientID) -> bool:
                pat_region_ids = self.patient(p).list_regions(region_ids=region_ids)
                if len(pat_region_ids) > 0:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'splits'.
        if splits != 'all':
            filepath = os.path.join(self.__path, 'holdout-split.csv')
            if not os.path.exists(filepath):
                raise ValueError(f"Holdout split file doesn't exist for dataset '{self}'.")
            split_df = load_files_csv(filepath)
            splits = arg_to_list(splits, Split, literals={ 'all': self.list_splits })
            def filter_fn(pat_id: PatientID) -> bool:
                pat_split = split_df[split_df['patient-id'] == pat_id].iloc[0]['split']
                if pat_split in splits:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'pat_ids'.
        if pat_ids != 'all':
            # Check for special group format.
            if isinstance(ids, str):
                regexp = r'^group:(\d+):(\d+)$'
                match = re.match(regexp, ids)
                if match:
                    group = int(match.group(1))
                    num_groups = int(match.group(2))
                    group_size = int(np.ceil(len(ids) / num_groups))
                    ids = ids[group * group_size:(group + 1) * group_size]
                else:
                    ids = arg_to_list(ids, PatientID)
            else:
                ids = arg_to_list(ids, PatientID)
            ids = [p for p in ids if p in pat_ids] 

        # Filter by 'exclude'.
        if exclude is not None:
            exclude = arg_to_list(exclude, PatientID)
            ids = [p for p in ids if p not in exclude]

        return ids

    def list_splits(self) -> List[Split]:
        filepath = os.path.join(self.__path, 'holdout-split.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"Holdout split file doesn't exist for dataset '{self}'.")
        split_df = load_files_csv(filepath)
        splits = list(sorted(split_df['split'].unique()))
        return splits

    def patient(
        self,
        id: Optional[PatientID] = None,
        n: Optional[int] = None,
        **kwargs) -> NiftiPatient:
        if n is not None:
            if id is not None:
                raise ValueError("Cannot specify both 'id' and 'n'.")
            id = self.list_patients()[n]

        # Filter indexes to include only rows relevant to the new patient.
        index = self.index[self.index['nifti-patient-id'] == str(id)].copy() if self.index is not None else None
        exc_df = self.excluded_labels[self.excluded_labels['patient-id'] == str(id)].copy() if self.excluded_labels is not None else None
        proc_df = self.processed_labels[self.processed_labels['patient-id'] == str(id)].copy() if self.processed_labels is not None else None

        return NiftiPatient(self, id, ct_from=self.__ct_from, index=index, excluded_labels=exc_df, processed_labels=proc_df, region_map=self.__region_map, **kwargs)

    def write_region(
        self,
        data: LabelData3D,
        pat_id: PatientID,
        study_id: StudyID,
        series_id: SeriesID,
        region: str,
        spacing: Spacing3D,
        offset: Point3D) -> None:
        filepath = os.path.join(self.__path, 'data', 'patients', pat_id, study_id, 'regions', series_id, f'{region}.nii.gz')
        save_nifti(data, filepath, spacing=spacing, offset=offset)
    
    def __load_index(self) -> None:
        filepath = os.path.join(self.path, 'index.csv')
        if os.path.exists(filepath):
            map_types = { 'dicom-patient-id': str, 'nifti-patient-id': str, 'study-id': str, 'series-id': str }
            self.__index = load_csv(filepath, map_types=map_types)
        else:
            self.__index = None
    
    def __load_excluded_labels(self) -> None:
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

    def __load_group_index(self) -> None:
        filepath = os.path.join(self.__path, 'group-index.csv')
        if os.path.exists(filepath):
            self.__group_index = read_csv(filepath).astype({ 'patient-id': str })
        else:
            self.__group_index = None

    # Copied from 'mymi/reporting/dataset/nift.py' to avoid circular dependency.
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
    
    def __load_processed_labels(self) -> None:
        filepath = os.path.join(self.__path, 'processed-labels.csv')
        if os.path.exists(filepath):
            self.__processed_labels = read_csv(filepath).astype({ 'patient-id': str })
            self.__processed_labels = self.__processed_labels.sort_values(['patient-id', 'region'])

            # Drop duplicates.
            dup_cols = ['patient-id', 'region']
            dup_df = self.__processed_labels[self.__processed_labels[dup_cols].duplicated()]
            if len(dup_df) > 0:
                logging.warning(f"Found {len(dup_df)} duplicate entries in 'processed-labels.csv', removing.")
                self.__processed_labels = self.__processed_labels[~self.__processed_labels[dup_cols].duplicated()]
        else:
            self.__processed_labels = None
    
    def __load_region_map(self) -> Optional[Dict[str, str]]:
        filepath = os.path.join(config.directories.config, 'region-mapping', f'{self.__global_id}.txt')
        if os.path.exists(filepath):
            region_map = {}
            with open(filepath, 'r') as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines]
                lines = [l for l in lines if l != '']
                for l in lines:
                    k, v = l.split(':')
                    region_map[k.strip()] = v.strip()
            return region_map    
        else:
            return None

    def __str__(self) -> str:
        return self.__global_id
    
# Add properties.
props = ['ct_from', 'global_id', 'id', 'path']
for p in props:
    setattr(NiftiDataset, p, property(lambda self, p=p: getattr(self, f'_{NiftiDataset.__name__}__{p}')))
