import os
import pydicom as dcm

from .rtstruct_series import RTSTRUCTSeries
from .dicom_series import DICOMModality, DICOMSeries

class RTPLANSeries(DICOMSeries):
    def __init__(
        self,
        study: 'DICOMStudy',
        id: str,
        load_ref_rtstruct: bool = True) -> None:
        self._global_id = f"{study} - {id}"
        self._study = study
        self._id = id
        self._path = os.path.join(study.path, 'rtplan', id)

        # Check that series exists.
        if not os.path.exists(self._path):
            raise ValueError(f"RTPLAN series '{self}' not found.")

        # Load reference CT series.
        if load_ref_rtstruct:
            rtplan = self.get_rtplan()
            rtstruct_id = rtplan.ReferencedStructureSetSequence[0].ReferencedSOPInstanceUID
            self._ref_rtstruct = RTSTRUCTSeries(study, rtstruct_id)

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
    def ref_rtstruct(self) -> str:
        return self._ref_rtstruct

    @property
    def study(self) -> str:
        return self._study

    def __str__(self) -> str:
        return self._global_id

    def get_rtplan(self) -> dcm.dataset.FileDataset:
        """
        returns: an RTPLAN DICOM object.
        """
        # Load RTPLAN.
        rtplans = os.listdir(self._path)
        rtplan = dcm.read_file(os.path.join(self._path, rtplans[0]))
        return rtplan
