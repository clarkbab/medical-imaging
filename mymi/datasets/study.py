from mymi.typing import *

from .region_map import RegionMap

class Study:
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        id: StudyID,
        ct_from: Optional['Study'],
        region_map: Optional[RegionMap] = None) -> None:
        self._dataset_id = str(dataset_id)
        self._pat_id = str(pat_id)
        self._id = str(id)
        self._ct_from = ct_from
        self._region_map = region_map

    @property
    def ct_from(self) -> Optional['Study']:
        return self._ct_from

    @property
    def dataset_id(self) -> DatasetID:
        return self._dataset_id

    @property
    def id(self) -> StudyID:
        return self._id

    @property
    def pat_id(self) -> PatientID:
        return self._pat_id

    @property
    def region_map(self) -> Optional[RegionMap]:
        return self._region_map

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        raise ValueError("Subclasses of 'Study' must implement '__str__' method.")
