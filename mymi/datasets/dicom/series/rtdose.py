import os
import pandas as pd
from typing import *

from mymi import config
from mymi.constants import *
from mymi.typing import *
from mymi.utils import *

from .rtplan import RtPlanSeries
from .series import DicomSeries

class RtDoseSeries(DicomSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID,
        index: pd.Series,
        index_policy: Dict[str, Any]) -> None:
        datasetpath = os.path.join(config.directories.datasets, 'dicom', dataset_id, 'data', 'patients')
        self._dataset_id = dataset_id
        self.__filepath = os.path.join(datasetpath, index['filepath'])
        self._id = id
        self.__index = index
        self.__index_policy = index_policy
        self.__modality = 'rtdose'
        self._pat_id = pat_id
        self._study_id = study_id

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

    def __load_data(self) -> None:
        self.__data, self.__spacing, self.__origin = from_rtdose_dicom(self.dicom)

# Add properties.
props = ['dataset_id', 'filepath', 'id', 'index', 'index_policy', 'modality', 'pat_id', 'ref_rtplan', 'study_id']
for p in props:
    setattr(RtDoseSeries, p, property(lambda self, p=p: getattr(self, f'_{RtDoseSeries.__name__}__{p}')))
