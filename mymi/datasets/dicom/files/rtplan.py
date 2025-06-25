import os
import pandas as pd
import pydicom as dcm
from pydicom.dataset import FileDataset

from mymi.typing import *

from .files import DicomFile

class RtPlanFile(DicomFile):
    def __init__(
        self,
        series: 'RtPlanSeries',
        id: DicomSOPInstanceUID):
        self.__global_id = f"{series}:{id}"
        self.__id = id
        self.__series = series

        # Get index.
        index = self.__series.index
        self.__index = index.loc[[self.__id]]
        self.__verify_index()
        self.__filepath = os.path.join(self.__series.study.patient.dataset.path, self.__index.iloc[0]['filepath'])

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> DicomSOPInstanceUID:
        return self.__id

    @property
    def filepath(self) -> str:
        return self.__filepath

    @property
    def series(self) -> str:
        return self.__series

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def dicom(self) -> FileDataset:
        return dcm.read_file(self.__filepath)

    def __str__(self) -> str:
        return self.__global_id

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RtPlan '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RtPlans found in index with DicomSOPInstanceUID '{self.__id}' for series '{self.__series}'.")
