from mymi.typing import *

class Series:
    def __init__(
        self,
        dataset: 'Dataset',
        pat: 'Patient',
        study: 'Study',
        id: SeriesID,
        config: Optional[Dict[str, Any]] = None) -> None:
        self._dataset = dataset
        self._config = config
        self._pat = pat
        self._study = study
        self._id = str(id)

    @property
    def dataset(self) -> 'Dataset':
        return self._dataset

    @property
    def id(self) -> SeriesID:
        return self._id

    @property
    def pat(self) -> 'Patient':
        return self._pat
    patient = pat

    @property
    def study(self) -> 'Study':
        return self._study

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
            study=self._study.id,
        )
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
