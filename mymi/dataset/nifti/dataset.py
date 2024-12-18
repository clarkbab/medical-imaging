import numpy as np
import os
from pandas import DataFrame, read_csv
import re
from typing import Dict, List, Literal, Optional, Union

from mymi import config
from mymi import logging
from mymi.regions import regions_to_list
from mymi.types import PatientID, PatientRegions

from ..dataset import Dataset, DatasetType
from ..shared import CT_FROM_REGEXP
from .patient import NiftiPatient

class NiftiDataset(Dataset):
    def __init__(
        self,
        name: str):
        # Create 'global ID'.
        self.__name = name
        self.__path = os.path.join(config.directories.datasets, 'nifti', self.__name)
        if not os.path.exists(self.__path):
            raise ValueError(f"Dataset 'NIFTI: {self.__name}' not found.")
        ct_from_name = None
        for f in os.listdir(self.__path):
            match = re.match(CT_FROM_REGEXP, f)
            if match:
                ct_from_name = match.group(1)
        self.__ct_from = NiftiDataset(ct_from_name) if ct_from_name is not None else None
        self.__global_id = f"NIFTI: {self.__name} (CT from - {self.__ct_from})" if self.__ct_from is not None else f"NIFTI: {self.__name}"

        self.__dicom_index = None                # Lazy-loaded.
        self.__excluded_labels = None           # Lazy-loaded.
        self.__group_index = None               # Lazy-loaded.
        self.__processed_labels = None          # Lazy-loaded.
        self.__region_map = self.__load_region_map()
        self.__loaded_dicom_index = False
        self.__loaded_excluded_labels = False
        self.__loaded_group_index = False
        self.__loaded_processed_labels = False

    @property
    def ct_from(self) -> Optional[Dataset]:
        return self.__ct_from
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dicom_index(self) -> Optional[DataFrame]:
        if not self.__loaded_dicom_index:
            self.__load_dicom_index()
            self.__loaded_dicom_index = True
        return self.__dicom_index

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
    def name(self) -> str:
        return self.__name
    
    @property
    def path(self) -> str:
        return self.__path

    @property
    def processed_labels(self) -> Optional[DataFrame]:
        if not self.__loaded_processed_labels:
            self.__load_processed_labels()
            self.__loaded_processed_labels = True
        return self.__processed_labels

    @property
    def region_map(self) -> Optional[Dict[str, str]]:
        return self.__region_map

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    def has_patient(
        self,
        pat_id: PatientID) -> bool:
        return pat_id in self.list_patients()

    def list_landmarks(self) -> List[str]:
        # Load each patient.
        landmarks = []
        pat_ids = self.list_patients()
        for pat_id in pat_ids:
            pat_landmarks = self.patient(pat_id).list_landmarks()
            landmarks += pat_landmarks
        landmarks = list(sorted([str(r) for r in np.unique(landmarks)]))
        return landmarks

    def list_patients(
        self,
        labels: Literal['included', 'excluded', 'all'] = 'included',
        regions: Optional[PatientRegions] = None) -> List[PatientID]:
        regions = regions_to_list(regions)

        if self.__ct_from is None:
            # Load patients from filenames.
            pat_path = os.path.join(self.__path, 'data', 'patients')
            pat_ids = list(sorted(os.listdir(pat_path)))
        else:
            # Load patients from 'ct_from' dataset.
            pat_ids = self.__ct_from.list_patients(labels=labels, regions=None)

        # Filter by 'regions'.
        if regions is not None:
            def filter_fn(pat_id: PatientID) -> bool:
                pat_regions = self.patient(pat_id).list_regions(labels=labels, only=regions)
                if len(pat_regions) > 0:
                    return True
                else:
                    return False
            pat_ids = list(filter(filter_fn, pat_ids))

        return pat_ids

    def list_regions(self) -> List[str]:
        # Load each patient.
        regions = []
        pat_ids = self.list_patients()
        for pat_id in pat_ids:
            pat_regions = self.patient(pat_id).list_regions()
            regions += pat_regions
        regions = list(sorted([str(r) for r in np.unique(regions)]))
        return regions

    def patient(
        self,
        id: PatientID,
        **kwargs) -> NiftiPatient:
        # Filte indexes to include only rows relevant to the new patient.
        dicom_index = self.dicom_index[self.dicom_index['nifti-patient-id'] == str(id)].iloc[0] if self.dicom_index is not None else None
        exc_df = self.excluded_labels[self.excluded_labels['patient-id'] == str(id)] if self.excluded_labels is not None else None
        proc_df = self.processed_labels[self.processed_labels['patient-id'] == str(id)] if self.processed_labels is not None else None

        return NiftiPatient(self, id, ct_from=self.__ct_from, dicom_index=dicom_index, excluded_labels=exc_df, processed_labels=proc_df, region_map=self.__region_map, **kwargs)
    
    def __load_dicom_index(self) -> None:
        filepath = os.path.join(config.directories.datasets, 'dicom', self.__name, 'index-nifti.csv')
        if os.path.exists(filepath):
            self.__dicom_index = read_csv(filepath).astype({ 'dicom-patient-id': str, 'nifti-patient-id': str })
        else:
            self.__dicom_index = None
    
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
        filepath = os.path.join(config.directories.config, 'region-mapping', f'NIFTI:{self.__name}.txt')
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
    