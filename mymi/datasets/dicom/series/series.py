from datetime import datetime as dt
from typing import *

from mymi.constants import *
from mymi.typing import *

from ...mixins import IndexMixin
from ...series import Series

DicomModality = Literal['ct', 'mr', 'rtdose', 'rtplan', 'rtstruct']

# Abstract class.
class DicomSeries(IndexMixin, Series):
    def __init__(
        self,
        modality: DicomModality,
        *args,
        **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._modality = modality

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
    def modality(self) -> DicomModality:
        return self._modality
 