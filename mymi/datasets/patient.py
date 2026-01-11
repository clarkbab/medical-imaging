from mymi.typing import *

from .region_map import RegionMap

class Patient:
    def __init__(
        self,
        dataset: 'Dataset',
        id: PatientID,
        config: Optional[Dict[str, Any]] = None,
        ct_from: Optional['Patient'] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self._dataset = dataset
        self._config = config
        self._id = str(id)
        self._ct_from = ct_from
        self._region_map = region_map

    @property
    def ct_from(self) -> Optional['Patient']:
        return self._ct_from

    @property
    def dataset(self) -> 'Dataset':
        return self._dataset

    @property
    def id(self) -> PatientID:
        return self._id

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
            dataset=self._dataset.id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from.id
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
