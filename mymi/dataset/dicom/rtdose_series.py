import os
import pandas as pd
import pydicom as dcm

from .rtplan_series import RTPLANSeries
from .dicom_series import DICOMModality, DICOMSeries

class RTDOSESeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str) -> None:
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id

        # Get index.
        index = self._study.index
        index = index[(index.modality == 'RTDOSE') & (index['series-id'] == id)]
        self._index = index
        self._path = index.iloc[0].filepath

        # Check that series exists.
        if len(index) == 0:
            raise ValueError(f"RTDOSE series '{self}' not found in index for study '{study}'.")

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTDOSE

    @property
    def index(self) -> pd.DataFrame:
        return self._index

    @property
    def study(self) -> str:
        return self._study

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return self._global_id

    def get_rtdose(self) -> dcm.dataset.FileDataset:
        filepath = self._index.iloc[0].filepath
        rtdose = dcm.read_file(filepath)
        return rtdose
