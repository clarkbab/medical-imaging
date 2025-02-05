from datetime import datetime
import pandas as pd
from typing import Any, List, Optional

from mymi import typing
from mymi.utils import append_row

from .study import DicomStudy, StudyInstanceUID
from .files import RegionMap
from .series import RtdoseSeries, RtplanSeries, RtstructSeries

class DicomPatient:
    def __init__(
        self,
        dataset: 'DicomDataset',
        id: typing.PatientID,
        ct_from: Optional['DicomPatient'] = None,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None,
        trimmed: bool = False):
        if trimmed:
            self.__global_id = f"{dataset} - {id} (trimmed)"
        else:
            self.__global_id = f"{dataset} - {id}"
        self.__ct_from = ct_from
        self.__default_rtdose = None        # Lazy-loaded.
        self.__default_rtplan = None        # Lazy-loaded.
        self.__default_rtstruct = None      # Lazy-loaded.
        self.__dataset = dataset
        self.__id = str(id)
        self.__region_dups = region_dups
        self.__region_map = region_map

        # Get patient index.
        index = self.__dataset.index
        index = index[index['patient-id'] == str(id)]
        self.__index = index

        # Get policies.
        self.__index_policy = self.__dataset.index_policy
        self.__region_policy = self.__dataset.region_policy

        # Check that patient ID exists.
        if len(index) == 0:
            raise ValueError(f"Patient '{self}' not found in index for dataset '{dataset}'.")

    @property
    def age(self) -> str:
        return getattr(self.get_cts()[0], 'PatientAge', '')

    @property
    def birth_date(self) -> str:
        return self.get_cts()[0].PatientBirthDate

    @property
    def ct_data(self):
        return self.default_rtstruct.ref_ct.data

    @property
    def ct_from(self) -> str:
        return self.__ct_from

    @property
    def ct_offset(self):
        return self.default_rtstruct.ref_ct.offset

    @property
    def ct_size(self):
        return self.default_rtstruct.ref_ct.size

    @property
    def ct_spacing(self):
        return self.default_rtstruct.ref_ct.spacing

    @property
    def dataset(self) -> str:
        return self.__dataset

    @property
    def default_rtdose(self) -> str:
        if self.__default_rtdose is None:
            self.__load_default_rtdose_and_rtplan()
        return self.__default_rtdose

    @property
    def default_rtplan(self) -> RtplanSeries:
        if self.__default_rtplan is None:
            self.__load_default_rtdose_and_rtplan()
        return self.__default_rtplan
    
    @property
    def default_rtstruct(self) -> RtstructSeries:
        if self.__default_rtstruct is None:
            self.__load_default_rtstruct()
        return self.__default_rtstruct
    
    @property
    def default_study(self) -> DicomStudy:
        # Choose the most recent study.
        study_id = self.list_studies()[-1]
        return self.study(study_id)

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dose_data(self):
        if self.default_rtdose is not None:
            return self.default_rtdose.data
        else:
            return None

    @property
    def dose_offset(self):
        if self.default_rtdose is not None:
            return self.default_rtdose.offset
        else:
            return None

    @property
    def dose_size(self):
        if self.default_rtdose is not None:
            return self.default_rtdose.size
        else:
            return None

    @property
    def dose_spacing(self):
        if self.default_rtdose is not None:
            return self.default_rtdose.spacing
        else:
            return None

    @property
    def id(self) -> str:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    @property
    def id(self) -> str:
        return self.__id

    @property
    def name(self) -> str:
        return self.get_cts()[0].PatientName

    @property
    def region_policy(self) -> pd.DataFrame:
        return self.__region_policy

    @property
    def sex(self) -> str:
        return self.get_cts()[0].PatientSex

    @property
    def size(self) -> str:
        return getattr(self.get_cts()[0], 'PatientSize', '')

    @property
    def study_date(self) -> datetime:
        return self.default_rtstruct.ref_ct.study_date

    @property
    def weight(self) -> str:
        return getattr(self.get_cts()[0], 'PatientWeight', '')

    def ct_slice_summary(self, *args, **kwargs):
        return self.default_rtstruct.ref_ct.slice_summary(*args, **kwargs)

    def ct_summary(self, *args, **kwargs):
        return self.default_rtstruct.ref_ct.summary(*args, **kwargs)

    @property
    def first_ct(self):
        return self.default_rtstruct.ref_ct.first_ct

    def get_rtdose(self, *args, **kwargs):
        return self.__default_rtdose_series.get_rtdose(*args, **kwargs)

    def has_landmark(self, *args, **kwargs):
        return self.default_rtstruct.has_landmark(*args, **kwargs)

    def has_regions(self, *args, **kwargs):
        return self.default_rtstruct.has_regions(*args, **kwargs)

    def info(self) -> pd.DataFrame:
        # Define dataframe structure.
        cols = {
            'age': str,
            'birth-date': str,
            'name': str,
            'sex': str,
            'size': str,
            'weight': str
        }
        df = pd.DataFrame(columns=cols.keys())

        # Add data.
        data = {}
        for col in cols.keys():
            col_method = col.replace('-', '_')
            data[col] = getattr(self, col_method)

        # Add row.
        df = append_row(df, data)

        # Set column types as 'append' crushes them.
        df = df.astype(cols)

        return df

    def landmark_data(self, *args, **kwargs):
        return self.default_rtstruct.landmark_data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs):
        return self.default_rtstruct.list_landmarks(*args, **kwargs)

    def list_studies(self) -> List[StudyInstanceUID]:
        # Sort studies by date/time - oldest first.
        study_ids = list(self.__index.sort_values(['study-date', 'study-time'])['study-id'].unique())
        return study_ids

    def list_regions(self, *args, **kwargs):
        return self.default_rtstruct.list_regions(*args, **kwargs)

    def region_data(self, *args, **kwargs):
        return self.default_rtstruct.region_data(*args, **kwargs)

    def region_summary(self, *args, **kwargs):
        return self.default_rtstruct.region_summary(*args, **kwargs)

    def study(
        self,
        id: str) -> DicomStudy:
        return DicomStudy(self, id, region_dups=self.__region_dups, region_map=self.__region_map)

    def __load_default_rtdose_and_rtplan(self) -> None:
        self.__default_rtplan = self.default_study.default_rtplan
        self.__default_rtdose = self.default_study.default_rtdose

    def __load_default_rtstruct(self) -> None:
        self.__default_rtstruct = self.default_study.default_rtstruct

    def __str__(self) -> str:
        return self.__global_id
