import numpy as np
import os

from mymi.geometry import fov
from mymi.typing import *
from mymi.utils import *

from ....dicom import DicomCtSeries, DicomDataset
from .image import NiftiImageSeries

class NiftiCtSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        study: StudyID,
        id: NiftiSeriesID,
        index: Optional[pd.DataFrame] = None,
        ) -> None:
        super().__init__('ct', dataset, pat, study, id, index=index)
        extensions = ['.nii', '.nii.gz', '.nrrd']
        basepath = os.path.join(config.directories.datasets, 'nifti', self._dataset_id, 'data', 'patients', self._pat_id, self._study_id, self._modality, self._id)
        filepath = None
        for e in extensions:
            fpath = f"{basepath}{e}"
            if os.path.exists(fpath):
                filepath = fpath
        if filepath is None:
            raise ValueError(f"No nifti ct series found for study '{self._study_id}'. Filepath: {basepath}, with extensions {extensions}.")
        self.__filepath = filepath

    @staticmethod
    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                if self.__filepath.endswith('.nii') or self.__filepath.endswith('.nii.gz'):
                    self.__data, self.__spacing, self.__origin = load_nifti(self.__filepath)
                elif self.__filepath.endswith('.nrrd'):
                    self.__data, self.__spacing, self.__origin = load_nrrd(self.__filepath)
                else:
                    raise ValueError(f'Unsupported file format: {self.__filepath}')
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> CtImageArray:
        return self.__data

    @property
    def dicom(self) -> DicomCtSeries:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset_id) & (index['patient-id'] == self._pat_id) & (index['study-id'] == self._study_id) & (index['series-id'] == self._id) & (index['modality'] == 'ct')].drop_duplicates()
        assert len(index) == 1
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).ct_series(row['dicom-series-id'])

    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Fov3D:
        return fov(self.__data, spacing=self.__spacing, origin=self.__origin, **kwargs)

    @property
    @ensure_loaded
    def origin(self) -> Point3D:
        return self.__origin

    @property
    @ensure_loaded
    def size(self) -> np.ndarray:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> np.ndarray:
        return self.__spacing

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)

# Add properties.
props = ['filepath']
for p in props:
    setattr(NiftiCtSeries, p, property(lambda self, p=p: getattr(self, f'_{NiftiCtSeries.__name__}__{p}')))
