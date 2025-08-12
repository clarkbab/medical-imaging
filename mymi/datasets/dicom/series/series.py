from datetime import datetime as dt
from typing import *

from mymi.constants import *
from mymi.typing import *

from ...mixins import IndexMixin

# Abstract class.
class DicomSeries(IndexMixin):
    @property
    def date(self) -> str:
        date_str = self.index['study-date']
        time_str = self.index['study-time']
        return f'{date_str}:{time_str}'

    @property
    def datetime(self) -> dt:
        parsed_dt = dt.strptime(self.date, f'{DICOM_DATE_FORMAT}:{DICOM_TIME_FORMAT}')
        return parsed_dt

    @property
    def global_id(self) -> str:
        return f'DICOM:{self.dataset_id}:{self.pat_id}:{self.study_id}:{self.id}[{self.modality}]'

    def __repr__(self) -> str:
        return self.global_id

    def __str__(self) -> str:
        return self.global_id
 