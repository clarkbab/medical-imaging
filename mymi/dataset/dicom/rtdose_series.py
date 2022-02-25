import os
import pydicom as dcm

from .rtplan_series import RTPLANSeries
from .dicom_series import DICOMModality, DICOMSeries

class RTDOSESeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str,
        load_ref_rtplan: bool = True) -> None:
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id
        self._path = os.path.join(study.path, 'rtdose', id)

        # Check that series exists.
        if not os.path.exists(self._path):
            raise ValueError(f"RTDOSE series '{self}' not found.")

        # Load reference CT series.
        if load_ref_rtplan:
            rtdose = self.get_rtdose()
            rtplan_id = rtdose.ReferenceRTPlanSequence[0].ReferencedSOPInstanceUID
            self._ref_rtplan = RTPLANSeries(study, rtplan_id)

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def modality(self) -> DICOMModality:
        return DICOMModality.RTSTRUCT

    @property
    def path(self) -> str:
        return self._path

    @property
    def ref_rtplan(self) -> str:
        return self._ref_rtplan

    @property
    def study(self) -> str:
        return self._study

    def __str__(self) -> str:
        return self._global_id

    def get_rtdose(self) -> dcm.dataset.FileDataset:
        """
        returns: an RTDOSE DICOM object.
        """
        # Load RTDOSE.
        rtdoses = os.listdir(self._path)
        rtdose = dcm.read_file(os.path.join(self._path, rtdoses[0]))
        return rtdose
