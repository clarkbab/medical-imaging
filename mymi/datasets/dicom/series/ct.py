from datetime import datetime
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import *

from mymi import config
from mymi.constants import DICOM_DATE_FORMAT, DICOM_TIME_FORMAT
from mymi.geometry import fov
from mymi.typing import *
from mymi.utils import *

from .series import DicomSeries

class CtSeries(DicomSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID,
        index: pd.DataFrame,
        index_policy: Dict[str, Any]) -> None:
        datasetpath = os.path.join(config.directories.datasets, 'dicom', dataset_id, 'data', 'patients')
        relpaths = list(index['filepath'])
        abspaths = [os.path.join(datasetpath, p) for p in relpaths]
        self._dataset_id = dataset_id
        self.__filepaths = abspaths
        self._id = id
        self.__index = index
        self.__index_policy = index_policy
        self.__modality = 'ct'
        self._pat_id = pat_id
        self._study_id = study_id

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper
    
    # Could return 'CTFile' objects - this would align with other series, but would create a lot of objects in memory.
    @property
    def dicoms(self) -> List[CtDicom]:
        # Sort CTs by z position, smallest first.
        ct_dicoms = [dcm.dcmread(f, force=False) for f in self.__filepaths]
        ct_dicoms = list(sorted(ct_dicoms, key=lambda c: c.ImagePositionPatient[2]))
        return ct_dicoms

    @property
    @ensure_loaded
    def data(self) -> np.ndarray:
        return self.__data

    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Fov3D:
        return fov(self.__data, spacing=self.__spacing, origin=self.__origin, **kwargs)

    @property
    def filepaths(self) -> List[str]:
        return self.__filepaths

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
        # Consistency is checked during indexing.
        # TODO: Change 'check_consistency' to be more granular and set based on the index policy.
        self.__data, self__spacing, self.__origin = from_ct_dicoms(self.dicoms, check_consistency=False)

# Add properties.
props = ['dataset_id', 'filepaths', 'id', 'index', 'index_policy', 'modality', 'pat_id', 'study_id']
for p in props:
    setattr(CtSeries, p, property(lambda self, p=p: getattr(self, f'_{CtSeries.__name__}__{p}')))
