import numpy as np
import os
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

from .study import NiftiStudy
from .images import CtNiftiImage

class NiftiPatient:
    def __init__(
        self,
        dataset: 'NiftiDataset',
        id: PatientID,
        check_path: bool = True,
        ct_from: Optional[str] = None,
        dicom_index: Optional[pd.Series] = None,
        excluded_labels: Optional[List[str]] = None,
        processed_labels: Optional[List[str]] = None,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__dataset = dataset
        self.__id = str(id)
        self.__global_id = f'{dataset}:{self.__id}'
        self.__dicom_index = dicom_index
        self.__region_map = region_map

        # Check that patient ID exists.
        self.__path = os.path.join(dataset.path, 'data', 'patients', self.__id)
        if check_path and not os.path.exists(self.__path):
            raise ValueError(f"Patient '{self}' not found. Filepath: '{self.__path}'.")

    def ct_extent(
        self,
        **kwargs) -> Optional[Union[ImageSizeMM3D, Size3D]]:
        return self.default_study.ct_extent(**kwargs)
    
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

    def has_image(self, *args, **kwargs) -> bool:
        return self.default_study.has_image(*args, **kwargs)

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

    def list_studies(
        self,
        study_ids: StudyIDs = 'all',    # Used for filtering.
        ) -> List[StudyID]:
        # Might have to deal with sorting at some point for 'default_study'.
        # Right now sorting is just alphabetical, which is fine if we're using anonymous IDs,
        # as they're sorted during DICOM -> NIFTI conversion.
        pat_study_ids = list(sorted(os.listdir(self.__path)))
        if study_ids != 'all':
            study_ids = arg_to_list(study_ids, StudyID)
            pat_study_ids = [s for s in pat_study_ids if s in study_ids]
        return pat_study_ids

    def region_images(self, *args, **kwargs) -> Dict[Region, RegionImage]:
        return self.default_study.region_images(*args, **kwargs)

    def study(
        self,
        id: StudyID,
        **kwargs) -> NiftiStudy:
        return NiftiStudy(self, id, region_map=self.__region_map, **kwargs)

    def __str__(self) -> str:
        return self.__global_id

# Add 'has_{mod}' properties.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiPatient, f'has_{m}', property(lambda self, m=m: getattr(self, f'default_{m}') is not None))

# Add default properties.
mods = ['ct', 'dose', 'landmarks', 'mr']
for m in mods:
    setattr(NiftiPatient, f'default_{m}', property(lambda self, m=m: getattr(self.default_study, f'default_{m}') if self.default_study is not None else None))

# Add image property shortcuts.
mods = ['ct', 'dose', 'mr']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiPatient, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(getattr(self, f'default_{m}'), p) if getattr(self, f'default_{m}') is not None else None))
