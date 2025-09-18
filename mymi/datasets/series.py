from mymi.typing import *

class Series:
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID) -> None:
        self._dataset_id = str(dataset_id)
        self._pat_id = str(pat_id)
        self._study_id = str(study_id)
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
    def study_id(self) -> StudyID:
        return self._study_id

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        raise ValueError("Subclasses of 'Series' must implement '__str__' method.")
