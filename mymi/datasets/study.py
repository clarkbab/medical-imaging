from mymi.typing import *

from .region_map import RegionMap

class Study:
    def __init__(
        self,
        dataset: 'Dataset',
        pat: 'Patient',
        id: StudyID,
        config: Optional[Dict[str, Any]] = None,
        ct_from: Optional['Study'] = None,
        region_map: Optional[RegionMap] = None) -> None:
        self._dataset = dataset
        self._config = config
        self._pat = pat
        self._id = str(id)
        self._ct_from = ct_from
        self._region_map = region_map

    @property
    def ct_from(self) -> Optional['Study']:
        return self._ct_from

    @property
    def dataset(self) -> 'Dataset':
        return self._dataset

    @property
    def id(self) -> StudyID:
        return self._id

    @property
    def pat(self) -> 'Patient':
        return self._pat
    patient = pat

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
            pat=self._pat.id,
        )
        if self._ct_from is not None:
            params['ct_from'] = self._ct_from
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
