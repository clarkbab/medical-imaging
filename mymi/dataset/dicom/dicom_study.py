import os
from typing import Dict, List, Optional

from .ct_series import CTSeries
from .dicom_series import DICOMModality, DICOMSeries
from .region_map import RegionMap
from .rtstruct_series import RTSTRUCTSeries

class DICOMStudy:
    def __init__(
        self,
        patient: 'DICOMPatient',
        id: str,
        region_map: Optional[RegionMap] = None):
        self._patient = patient
        self._id = id
        self._region_map = region_map
        self._global_id = f"{patient} - {id}"
        self._path = os.path.join(patient.path, id)
    
        # Check that study exists.
        if not os.path.isdir(self._path):
            raise ValueError(f"DICOM study '{self}' not found.")

    @property
    def description(self) -> str:
        return self._global_id

    @property
    def id(self) -> str:
        return self._id

    def __str__(self) -> str:
        return self._global_id

    @property
    def path(self) -> str:
        return self._path

    @property
    def patient(self) -> str:
        return self._patient

    def list_series(
        self,
        modality: str) -> List[str]:
        # Get modality folder.
        if not modality in ('ct', 'rtstruct'):
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")

        # Return series.
        path = os.path.join(self._path, modality)
        if os.path.exists(path):
            return list(sorted(os.listdir(path)))
        else:
            return []

    def get_series(
        self,
        id: str,
        modality: str,
        **kwargs: Dict) -> DICOMSeries:
        if modality == 'ct':
            return CTSeries(self, id, region_map=self._region_map, **kwargs)
        elif modality == 'rtstruct':
            return RTSTRUCTSeries(self, id, region_map=self._region_map, **kwargs)
        else:
            raise ValueError(f"Unrecognised DICOM modality '{modality}'.")
