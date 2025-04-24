import numpy as np
import os
from typing import *

from mymi.typing import *

from .data import *

class NiftiStudy:
    def __init__(
        self,
        patient: 'NiftiPatient',
        id: StudyID,
        check_path: bool = True,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__global_id = f'{patient}:{id}'
        self.__id = id
        self.__path = os.path.join(patient.path, id)
        self.__patient = patient
        self.__region_map = region_map

        # Check that study ID exists.
        self.__path = os.path.join(patient.path, self.__id)
        if check_path and not os.path.exists(self.__path):
            raise ValueError(f"Study '{self}' not found. Filepath: '{self.__path}'.")

    @property
    def ct_data(self) -> Optional[CtImage]:
        def_ct = self.default_ct
        if def_ct is None:
            return None
        return def_ct.data

    @property
    def ct_affine(self) -> Optional[np.ndarray]:
        def_ct = self.default_ct
        if def_ct is None:
            return None
        return def_ct.affine

    @property
    def ct_offset(self) -> Optional[Point3D]:
        def_ct = self.default_ct
        if def_ct is None:
            return None
        return def_ct.offset

    @property
    def ct_path(self) -> str:
        return self.default_ct.path

    @property
    def ct_size(self) -> Optional[ImageSize3D]:
        def_ct = self.default_ct
        if def_ct is None:
            return None
        return def_ct.size

    @property
    def ct_spacing(self) -> Optional[ImageSpacing3D]:
        def_ct = self.default_ct
        if def_ct is None:
            return None
        return def_ct.spacing

    @property
    def default_ct(self) -> Optional[CtData]:
        data_ids = self.list_data('CT')
        if len(data_ids) == 0:
            return None
        else:
            return CtData(self, data_ids[-1])

    @property
    def default_landmarks(self) -> Optional[LandmarkData]:
        data_ids = self.list_data('LANDMARKS')
        if len(data_ids) == 0:
            return None
        else:
            return LandmarkData(self, data_ids[-1])

    @property
    def default_regions(self) -> Optional[RegionData]:
        data_ids = self.list_data('REGIONS')
        if len(data_ids) == 0:
            return None
        else:
            return RegionData(self, data_ids[-1], region_map=self.__region_map)

    @property
    def id(self) -> str:
        return self.__id

    @property
    def path(self) -> str:
        return self.__path

    @property
    def patient(self) -> 'NiftiPatient':
        return self.__patient

    def has_data(
        self,
        id: str,
        modality: Modality) -> bool:
        return id in self.list_data(modality)

    def has_landmark(self, *args, **kwargs) -> bool:
        lms = self.default_landmarks
        if lms is None:
            return False
        else:
            return lms.has_landmark(*args, **kwargs)

    def has_regions(self, *args, **kwargs) -> bool:
        def_regions = self.default_regions 
        if def_regions is None:
            return False
        return def_regions.has_regions(*args, **kwargs)

    def landmark_data(self, *args, **kwargs) -> Optional[Landmarks]:
        def_landmarks = self.default_landmarks
        if def_landmarks is None:
            return None
        return def_landmarks.data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs) -> List[Landmark]:
        def_landmarks = self.default_landmarks
        if def_landmarks is None:
            return []
        return def_landmarks.list_landmarks(*args, **kwargs)

    def list_regions(self, *args, **kwargs) -> List[Region]:
        def_regions = self.default_regions 
        if def_regions is None:
            return []
        return def_regions.list_regions(*args, **kwargs)

    def list_data(
        self,
        modality: Modality) -> List[str]:
        if modality == 'CT':
            filepath = os.path.join(self.__path, 'ct')
            if os.path.exists(filepath):
                data_ids = list(sorted(os.listdir(filepath)))
                data_ids = [s.replace('.nii.gz', '') for s in data_ids]
            else:
                data_ids = []
        elif modality == 'DOSE':
            pass
        elif modality == 'LANDMARKS':
            filepath  = os.path.join(self.__path, 'landmarks')
            if os.path.exists(filepath):
                data_ids = list(sorted(os.listdir(filepath)))
                data_ids = [s.replace('.csv', '') for s in data_ids]
            else:
                data_ids = []
        elif modality == 'MR':
            filepath = os.path.join(self.__path, 'mr')
            if os.path.exists(filepath):
                data_ids = list(sorted(os.listdir(filepath)))
                data_ids = [s.replace('.nii.gz', '') for s in data_ids]
            else:
                data_ids = []
        elif modality == 'REGIONS':
            filepath = os.path.join(self.__path, 'regions')
            if os.path.exists(filepath):
                data_ids = list(sorted(os.listdir(filepath)))
            else:
                data_ids = []
        else:
            raise ValueError(f"Modality '{modality}' not supported.")
    
        return data_ids

    def region_data(self, *args, **kwargs) -> Optional[RegionData]:
        def_regions = self.default_regions 
        if def_regions is None:
            return None
        return def_regions.data(*args, **kwargs)

    def region_path(self, *args, **kwargs) -> Optional[str]:
        def_regions = self.default_regions 
        if def_regions is None:
            return None
        return def_regions.region_path(*args, **kwargs)

    def data(
        self,
        id: str,
        modality: Optional[Modality] = None) -> NiftiData:
        if modality is None:
            modality = self.data_modality(id)

        if modality == 'CT':
            data = CtData(self, id)
        elif modality == 'DOSE':
            pass
        elif modality == 'LANDMARKS':
            data = LandmarkData(self, id)
        elif modality == 'MR':
            data = MrData(self, id)
        elif modality == 'REGIONS':
            data = RegionData(self, id, region_map=self.__region_map)
        else:
            raise ValueError(f"Modality '{modality}' not supported.")

        return data

    def data_modality(
        self,
        id: SeriesID) -> Modality:
        modalities = ['CT', 'MR', 'LANDMARKS', 'REGIONS']
        for m in modalities:
            data_ids = self.list_data(m)
            if id in data_ids:
                return m
        raise ValueError(f"Data with ID '{id}' not found in study '{self.__global_id}'.")

    def __str__(self) -> str:
        return self.__global_id
    