import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional

from mymi.typing import ImageSize3D, ImageSpacing3D, Landmarks, Landmark, Region, PointMM3D, SeriesID, StudyID

from .data import CtData, LandmarkData, Modality, NiftiData, RegionData

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
    def ct_data(self) -> np.ndarray:
        return self.default_ct.data

    @property
    def ct_affine(self) -> np.ndarray:
        return self.default_ct.affine

    @property
    def ct_offset(self) -> PointMM3D:
        return self.default_ct.offset

    @property
    def ct_path(self) -> str:
        return self.default_ct.path

    @property
    def ct_size(self) -> ImageSize3D:
        return self.default_ct.size

    @property
    def ct_spacing(self) -> ImageSpacing3D:
        return self.default_ct.spacing

    @property
    def default_ct(self) -> Optional[CtData]:
        data_ids = self.list_data(Modality.CT)
        if len(data_ids) == 0:
            return None
        else:
            return CtData(self, data_ids[-1])

    @property
    def default_landmarks(self) -> Optional[LandmarkData]:
        data_ids = self.list_data(Modality.LANDMARKS)
        if len(data_ids) == 0:
            return None
        else:
            return LandmarkData(self, data_ids[-1])

    @property
    def default_regions(self) -> Optional[RegionData]:
        data_ids = self.list_data(Modality.REGIONS)
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
        if modality == Modality.CT:
            ct_path = os.path.join(self.__path, 'ct')
            if os.path.exists(ct_path):
                data_ids = list(sorted(os.listdir(ct_path)))
                data_ids = [s.replace('.nii.gz', '') for s in data_ids]
            else:
                data_ids = []
        elif modality == Modality.LANDMARKS:
            landmarks_path  = os.path.join(self.__path, 'landmarks')
            if os.path.exists(landmarks_path):
                data_ids = list(sorted(os.listdir(landmarks_path)))
                data_ids = [s.replace('.csv', '') for s in data_ids]
            else:
                data_ids = []
        elif modality == Modality.REGIONS:
            regions_path = os.path.join(self.__path, 'regions')
            if os.path.exists(regions_path):
                data_ids = list(sorted(os.listdir(regions_path)))
            else:
                data_ids = []
        elif modality == Modality.DOSE:
            pass
    
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
        modality: Modality) -> NiftiData:
        if modality == Modality.CT:
            data = CtData(self, id)
        elif modality == Modality.LANDMARKS:
            data = LandmarkData(self, id)
        elif modality == Modality.REGIONS:
            data = RegionData(self, id, region_map=self.__region_map)
        elif modality == Modality.DOSE:
            pass

        return data

    def __str__(self) -> str:
        return self.__global_id
    