import os
import pandas as pd
from typing import *

from mymi import config
from mymi.typing import *

from .series import DicomSeries

class RtPlanSeries(DicomSeries):
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
        self._index = index
        self._index_policy = index_policy
        self.__modality = 'rtplan'
        self._pat_id = pat_id
        self._study_id = study_id

    @property
    def dicom(self) -> RtPlanDicom:
        return dcm.dcmread(self.__filepath)

# Add properties.
props = ['dataset_id', 'filepath', 'id', 'index', 'index_policy', 'modality', 'pat_id', 'ref_rtstruct', 'study_id']
for p in props:
    setattr(RtPlanSeries, p, property(lambda self, p=p: getattr(self, f'_{RtPlanSeries.__name__}__{p}')))


