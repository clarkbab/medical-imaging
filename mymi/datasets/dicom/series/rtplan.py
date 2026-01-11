import os
import pandas as pd
from typing import *

from mymi import config
from mymi.typing import *

from .series import DicomSeries

class DicomRtPlanSeries(DicomSeries):
    def __init__(
        self,
        dataset: 'DicomDataset',
        pat: 'DicomPatient',
        study: 'DicomStudy',
        id: SeriesID,
        index: pd.Series,
        index_policy: Dict[str, Any]) -> None:
        super().__init__('rtplan', dataset, pat, study, id, index=index, index_policy=index_policy)
        dspath = os.path.join(config.directories.datasets, 'dicom', self._dataset.id, 'data', 'patients')
        self.__filepath = os.path.join(dspath, index['filepath'])

    @property
    def dicom(self) -> RtPlanDicom:
        return dcm.dcmread(self.__filepath)

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath', 'ref_rtstruct']
for p in props:
    setattr(DicomRtPlanSeries, p, property(lambda self, p=p: getattr(self, f'_{DicomRtPlanSeries.__name__}__{p}')))


