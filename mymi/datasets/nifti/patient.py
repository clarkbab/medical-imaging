import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional

from mymi.typing import ImageSpacing3D, Landmarks, PatientID, Landmark, Region, Point3D, StudyID

from .study import NiftiStudy

class NiftiPatient:
    def __init__(
        self,
        dataset: 'NiftiDataset',
        id: PatientID,
        check_path: bool = True,
        ct_from: Optional['NiftiDataset'] = None,
        dicom_index: Optional[pd.Series] = None,
        excluded_labels: Optional[pd.DataFrame] = None,
        processed_labels: Optional[pd.DataFrame] = None,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__dataset = dataset
        self.__ct_from = ct_from
        self.__id = str(id)
        self.__excluded_labels = excluded_labels
        self.__global_id = f'{dataset}:{self.__id}'
        self.__dicom_index = dicom_index
        self.__processed_labels = processed_labels
        self.__region_map = region_map
        self.__inverse_region_map  = {v: k for k, v in self.__region_map.items()} if self.__region_map is not None else None

        # Check that patient ID exists.
        self.__path = os.path.join(dataset.path, 'data', 'patients', self.__id)
        if check_path and not os.path.exists(self.__path):
            raise ValueError(f"Patient '{self}' not found. Filepath: '{self.__path}'.")

    @property
    def ct_data(self) -> np.ndarray:
        return self.default_study.ct_data

    @property
    def ct_affine(self) -> np.ndarray:
        return self.default_study.ct_affine

    @property
    def ct_offset(self) -> Point3D:
        return self.default_study.ct_offset

    @property
    def ct_path(self) -> str:
        return self.default_study.ct_path

    @property
    def ct_size(self) -> np.ndarray:
        return self.default_study.ct_size

    @property
    def ct_spacing(self) -> ImageSpacing3D:
        return self.default_study.ct_spacing
    
    @property
    def dataset(self) -> 'NiftiDataset':
        return self.__dataset

    @property
    def default_study(self) -> Optional[NiftiStudy]:
        study_ids = self.list_studies()
        if len(study_ids) == 0:
            return None
        else:
            return NiftiStudy(self, study_ids[-1])
    
    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dicom_index(self) -> pd.DataFrame:
        return self.__dicom_index

    @property
    def dose_data(self) -> np.ndarray:
        return self.default_study.default_rtdose.data

    @property
    def has_dose(self) -> bool:
        pass

    @property
    def id(self) -> str:
        return self.__id

    @property
    def origin(self) -> Dict[str, str]:
        if self.dicom_index is None:
            raise ValueError(f"No 'index-nifti.csv' provided for DICOM dataset '{self.__dataset.name}'.")
        info = self.dicom_index.to_dict()
        return info

    @property
    def path(self) -> str:
        return self.__path

    def has_data(self, *args, **kwargs) -> bool:
        return self.default_study.has_data(*args, **kwargs)

    def has_landmark(self, *args, **kwargs) -> bool:
        return self.default_study.has_landmark(*args, **kwargs)

    def has_regions(self, *args, **kwargs) -> bool:
        return self.default_study.has_regions(*args, **kwargs)

    def landmark_data(self, *args, **kwargs) -> Landmarks:
        return self.default_study.landmark_data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs) -> List[Landmark]:
        return self.default_study.list_landmarks(*args, **kwargs)

    def list_regions(self, *args, **kwargs) -> List[Region]:
        return self.default_study.list_regions(*args, **kwargs)

    def list_studies(self) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        return list(sorted(os.listdir(self.__path)))

    def region_data(self, *args, **kwargs) -> Dict[Region, np.ndarray]:
        return self.default_study.region_data(*args, **kwargs)

    def study(
        self,
        id: StudyID,
        **kwargs) -> NiftiStudy:
        return NiftiStudy(self, id, region_map=self.__region_map, **kwargs)

    def __str__(self) -> str:
        return self.__global_id
