from mymi.typing import *

from .region_map import RegionMap

class Patient:
    def __init__(
        self,
        dataset_id: DatasetID,
        id: PatientID,
        ct_from: Optional['Patient'] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self._dataset_id = str(dataset_id)
        self._id = str(id)
        self._ct_from = ct_from
        self._region_map = region_map

    @property
    def ct_from(self) -> Optional['Patient']:
        return self._ct_from

    @property
    def dataset_id(self) -> DatasetID:
        return self._dataset_id

    @property
    def id(self) -> PatientID:
        return self._id

    @property
    def region_map(self) -> Optional[RegionMap]:
        return self._region_map

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        raise ValueError("Subclasses of 'Patient' must implement '__str__' method.")
