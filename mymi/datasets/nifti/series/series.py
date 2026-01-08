from typing import *

from ...dicom import DicomDataset, DicomSeries
from ...mixins import IndexMixin
from ...series import Series

NiftiModality = Literal['ct', 'dose', 'landmarks', 'mr', 'plan', 'regions']

NIFTI_DICOM_MODALITY_MAP = dict(
    ct='ct',
    dose='rtdose',
    landmarks='rtstruct',
    mr='mr',
    plan='rtplan',
    regions='rtstruct',
)

class NiftiSeries(IndexMixin, Series):
    def __init__(
        self,
        modality: NiftiModality,
        *args,
        **kwargs) -> None:
        self._modality = modality
        self.__dicom_modality = NIFTI_DICOM_MODALITY_MAP[self._modality]
        super().__init__(*args, **kwargs)

    @property
    def date(self) -> Optional[str]:
        # May implement in dicom -> nifti processing in future.
        return None

    @property
    def dicom(self) -> DicomSeries:
        raise ValueError("Subclasses of 'NiftiSeries' must implement 'dicom' method.")

    @property
    def modality(self) -> NiftiModality:
        return self._modality
