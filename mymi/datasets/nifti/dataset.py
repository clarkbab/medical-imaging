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
from ..dicom import DicomDataset
from ..mixins import IndexMixin
from ..region_map import RegionMap
from .patient import NiftiPatient

class NiftiDataset(IndexMixin, Dataset):
    def __init__(
        self,
        id: DatasetID) -> None:
        self._path = os.path.join(config.directories.datasets, 'nifti', str(id))
        if not os.path.exists(self._path):
            raise ValueError(f"No nifti dataset '{id}' found at path: {self._path}")
        ct_from = None
        for f in os.listdir(self._path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from = match.group(1)
        ct_from = NiftiDataset(ct_from) if ct_from is not None else None
        super().__init__(id, ct_from=ct_from)

    @property
    def dicom(self) -> DicomDataset:
        ds = self._index['dicom-dataset'].unique().tolist()
        assert len(ds) == 1
        return DicomDataset(ds[0])

    def has_patient(
        self,
        pat: PatientID = 'all',
        all: bool = False) -> bool:
        all_pats = self.list_patients()
        subset_pats = self.list_patients(pat=pat)
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

    @Dataset.ensure_loaded
    def list_patients(
        self,
        exclude: Optional[PatientIDs] = None,
        group: PatientGroups = 'all',
        pat: PatientIDs = 'all',    # Saves on filtering code elsewhere.
        region: RegionIDs = 'all',
        ) -> List[PatientID]:
        # Load patients from filenames.
        dirpath = os.path.join(self._path, 'data', 'patients')
        all_ids = list(sorted(os.listdir(dirpath)))

        # Filter by 'region'.
        ids = all_ids.copy()
        if region != 'all':
            regions = regions_to_list(region)
            all_regions = self.list_regions()

            # # Check that 'regions' are valid.
            # for r in regions:
            #     if not r in all_regions:
            #         logging.warning(f"Filtering by region '{r}' but it doesn't exist in dataset '{self}'.")

            def filter_fn(p: PatientID) -> bool:
                pat_regions = self.patient(p).list_regions(region=regions)
                if len(pat_regions) > 0:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'group'.
        if group != 'all':
            if self._groups is None:
                raise ValueError(f"File 'groups.csv' not found for dicom dataset '{self._id}'.")
            all_groups = self.list_groups()
            groups = arg_to_list(group, PatientGroup, literals={ 'all': all_groups })
            for g in groups:
                if g not in all_groups:
                    raise ValueError(f"Group {g} not found.")

            def filter_fn(p: PatientID) -> bool:
                pat_groups = self._groups[self._groups['patient-id'] == p]
                if len(pat_groups) == 0:
                    return False
                elif len(pat_groups) > 1:
                    raise ValueError(f"Patient {p} is a member of more than one group.")
                pat_group = pat_groups.iloc[0]['group-id']
                if pat_group in groups:
                    return True
                else:
                    return False
            ids = list(filter(filter_fn, ids))

        # Filter by 'exclude'.
        if exclude is not None:
            exclude = arg_to_list(exclude, PatientID)
            ids = [p for p in ids if p not in exclude]

        # Filter by 'pat'.
        if pat != 'all':
            pat_ids = arg_to_list(pat, PatientID)

            # # Check that 'pat_ids' are valid.
            # for p in pat_ids:
            #     if not p in all_ids:
            #         logging.warning(f"Filtering by patient ID '{p}' but it doesn't exist in dataset '{self}'.")

            filt_ids = []
            for i, id in enumerate(ids):
                # Check if any of the passed 'pat_ids' references this ID.
                for j, pid in enumerate(pat_ids):
                    if pid.startswith('i:'):
                        if '-' in pid and not 'i:-' in pid:   # Make sure negative indexing doesn't match - probably better with a regexp.
                            # Format 'i:4-8'.
                            min_idx, max_idx = pid.split(':')[1].split('-')
                            min_idx, max_idx = int(min_idx), int(max_idx)
                            if i >= min_idx and i < max_idx:
                                filt_ids.append(id)
                                break
                        else:
                            # Format: 'i:4' or 'i:-4'.
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
        pat: PatientIDs = 'all',
        region: RegionIDs = 'all',  # Used for filtration.
        ) -> List[RegionID]:
        # Load all patients.
        pat_ids = self.list_patients(pat=pat)

        # Trawl the depths for region IDs.
        ids = []
        for p in pat_ids:
            pat = self.patient(p)
            study_ids = pat.list_studies()
            for s in study_ids:
                study = pat.study(s)
                series_ids = study.list_regions_series()
                for s in series_ids:
                    series = study.regions_series(s)
                    ids += series.list_regions(region=region)
        ids = list(str(i) for i in np.unique(ids))

        return ids

    def _load_data(self) -> None:
        # Load groups.
        filepath = os.path.join(self._path, 'groups.csv')
        self._groups = load_csv(filepath) if os.path.exists(filepath) else None

        # Load index.
        filepath = os.path.join(self._path, 'index.csv')
        if os.path.exists(filepath):
            map_types = { 'dicom-patient-id': str, 'patient-id': str, 'study-id': str, 'series-id': str }
            self._index = load_csv(filepath, map_types=map_types)
        else:
            self._index = None
    
        # Load excluded labels.
        filepath = os.path.join(self._path, 'excluded-labels.csv')
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
        filepath = os.path.join(self._path, 'group-index.csv')
        if os.path.exists(filepath):
            self.__group_index = read_csv(filepath).astype({ 'patient-id': str })
        else:
            self.__group_index = None

        # Load region map.
        filepath = os.path.join(self._path, 'region-map.yaml')
        if os.path.exists(filepath):
            self.__region_map = RegionMap(load_yaml(filepath))
        else:
            self.__region_map = None

    # Copied from 'mymi/reports/dataset/nift.py' to avoid circular dependency.
    def __load_patient_regions_report(
        self,
        exists_only: bool = False) -> Union[DataFrame, bool]:
        filepath = os.path.join(self._path, 'reports', 'region-count.csv')
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
    def n_patients(self) -> int:
        return len(self.list_patients())

    @Dataset.ensure_loaded
    def patient(
        self,
        id: Optional[PatientID] = None,
        group: PatientGroups = 'all',
        n: Optional[int] = None,
        **kwargs) -> NiftiPatient:
        id = handle_idx_prefix(id, lambda: self.list_patients(group=group))
        if n is not None:
            if id is not None:
                raise ValueError("Cannot specify both 'id' and 'n'.")
            id = self.list_patients()[n]

        # Filter indexes to include only rows relevant to the new patient.
        index = self._index[self._index['patient-id'] == str(id)].copy() if self._index is not None else None
        exc_df = self.__excluded_labels[self.__excluded_labels['patient-id'] == str(id)].copy() if self.__excluded_labels is not None else None

        # Get 'ct_from' patient.
        if self._ct_from is not None and self._ct_from.has_patient(id):
            ct_from = self._ct_from.patient(id)
        else:
            ct_from = None

        return NiftiPatient(self, id, ct_from=ct_from, index=index, excluded_labels=exc_df, region_map=self.__region_map, **kwargs)

    @property
    @Dataset.ensure_loaded
    def region_map(self) -> Optional[RegionMap]:
        return self.__region_map

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI
