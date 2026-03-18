from mymi.typing import *

from .regions_map import RegionsMap

class Patient:
    def __init__(
        self,
        dataset: 'Dataset',
        id: PatientID,
        config: Optional[Dict[str, Any]] = None,
        ct_from: Optional['Patient'] = None,
        regions_map: Optional[RegionsMap] = None) -> None:
        self._dataset = dataset
        self._config = config
        self._id = str(id)
        self._ct_from = ct_from
        self._regions_map = regions_map

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
    def regions_map(self) -> Optional[RegionsMap]:
        return self._regions_map

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
