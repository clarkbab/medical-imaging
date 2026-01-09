from mymi.typing import *

class Series:
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        study: StudyID,
        id: SeriesID,
        config: Optional[Dict[str, Any]] = None) -> None:
        self._dataset_id = str(dataset)
        self._config = config
        self._pat_id = str(pat)
        self._study_id = str(study)
        self._id = str(id)

    @property
    def dataset_id(self) -> DatasetID:
        return self._dataset_id

    @property
    def id(self) -> SeriesID:
        return self._id

    @property
    def pat_id(self) -> PatientID:
        return self._pat_id

    @property
    def study(self) -> StudyID:
        return self._study_id

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
            study=self._study_id,
        )
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
