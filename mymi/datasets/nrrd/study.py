import numpy as np
import os
from typing import *

from mymi.typing import *

from .data import *

class NrrdStudy:
    def __init__(
        self,
        patient: 'NrrdPatient',
        id: StudyID,
        check_path: bool = True,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__global_id = f'{patient}:{id}'
        self.__id = id
        self.__path = os.path.join(patient.path, self.__id)
        if check_path and not os.path.exists(self.__path):
            raise ValueError(f"Study '{self}' not found. Filepath: '{self.__path}'.")
        self.__patient = patient
        self.__region_map = region_map

    @property
    def ct_data(self) -> np.ndarray:
        return self.default_ct.data

    @property
    def ct_offset(self) -> Point3D:
        return self.default_ct.offset

    @property
    def ct_path(self) -> str:
        return self.default_ct.path

    @property
    def ct_size(self) -> Size3D:
        return self.default_ct.size

    @property
    def ct_spacing(self) -> Spacing3D:
        return self.default_ct.spacing

    @property
    def default_ct(self) -> Optional[CtData]:
        data_ids = self.list_data('ct')
        if len(data_ids) == 0:
            return None
        else:
            return CtData(self, data_ids[-1])

    @property
    def default_landmarks(self) -> Optional[LandmarksData]:
        data_ids = self.list_data('landmarks')
        if len(data_ids) == 0:
            return None
        else:
            return LandmarksData(self, data_ids[-1])

    @property
    def default_regions(self) -> Optional[RegionsData]:
        data_ids = self.list_data('regions')
        if len(data_ids) == 0:
            return None
        else:
            return RegionsData(self, data_ids[-1], region_map=self.__region_map)

    @property
    def id(self) -> str:
        return self.__id

    @property
    def path(self) -> str:
        return self.__path

    @property
    def patient(self) -> 'NrrdPatient':
        return self.__patient

    def has_data(
        self,
        id: str,
        modality: NrrdModality) -> bool:
        return id in self.list_data(modality)

    def landmarks_data(self, *args, **kwargs) -> Landmarks:
        return self.default_landmarks.data(*args, **kwargs)

    def list_data(
        self,
        modality: NrrdModality) -> List[str]:
        if modality == 'ct':
            data_ids = list(sorted(os.listdir(os.path.join(self.__path, 'ct'))))
            data_ids = [s.replace('.nrrd', '') for s in data_ids]
        elif modality == 'landmarks':
            data_ids = list(sorted(os.listdir(os.path.join(self.__path, 'landmarks'))))
            data_ids = [s.replace('.csv', '') for s in data_ids]
        elif modality == 'regions':
            data_ids = list(sorted(os.listdir(os.path.join(self.__path, 'regions'))))
        elif modality == 'dose':
            pass
        else:
            raise ValueError(f"Modality '{modality}' not recognised.")
    
        return data_ids

    def regions_data(self, *args, **kwargs) -> Dict[Region, np.ndarray]:
        return self.default_regions.data(*args, **kwargs)

    def region_path(self, *args, **kwargs) -> str:
        return self.default_regions.region_path(*args, **kwargs)

    def data(
        self,
        id: str,
        modality: NrrdModality) -> NrrdData:
        if modality == 'ct':
            data = CtData(self, id)
        elif modality == 'landmarks':
            data = LandmarksData(self, id)
        elif modality == 'regions':
            data = RegionsData(self, id, region_map=self.__region_map)
        elif modality == 'dose':
            pass

        return data

    def __str__(self) -> str:
        return self.__global_id
    