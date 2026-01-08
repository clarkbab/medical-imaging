from mymi.typing import *

from .region_map import RegionMap

class Study:
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        id: StudyID,
        ct_from: Optional['Study'],
        region_map: Optional[RegionMap] = None) -> None:
        self._dataset_id = str(dataset)
        self._pat_id = str(pat)
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

    def __str__(
        self,
        class_name: str,
        ) -> str:
        params = dict(
            id=self._id,
            dataset=self._dataset_id,
            pat=self._pat_id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
