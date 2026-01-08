import os
import pandas as pd
from typing import *

from mymi import config
from mymi.constants import *
from mymi.typing import *
from mymi.utils import *

from .rtplan import DicomRtPlanSeries
from .series import DicomSeries

class DicomRtDoseSeries(DicomSeries):
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        study: StudyID,
        id: SeriesID,
        index: pd.Series,
        index_policy: Dict[str, Any]) -> None:
        super().__init__('rtdose', dataset, pat, study, id, index=index, index_policy=index_policy)
        dspath = os.path.join(config.directories.datasets, 'dicom', self._dataset_id, 'data', 'patients')
        self.__filepath = os.path.join(dspath, index['filepath'])

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> DoseImageArray:
        return self.__data

    @property
    def dicom(self) -> RtDoseDicom:
        return dcm.dcmread(self.__filepath)

    def __load_data(self) -> None:
        self.__data, self.__spacing, self.__origin = from_rtdose_dicom(self.dicom)

    @property
    @ensure_loaded
    def origin(self) -> Point3D:
        return self.__origin

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath', 'ref_rtplan']
for p in props:
    setattr(DicomRtDoseSeries, p, property(lambda self, p=p: getattr(self, f'_{DicomRtDoseSeries.__name__}__{p}')))
