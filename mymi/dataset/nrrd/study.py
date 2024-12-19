import numpy as np
import os
import pandas as pd
from typing import Dict, List, Optional

from mymi.types import ImageSize3D, ImageSpacing3D, Landmarks, PatientLandmark, PatientRegion, PointMM3D, SeriesID, StudyID

from .data import CtData, LandmarkData, Modality, NrrdData, RegionData

class NrrdStudy:
    def __init__(
        self,
        patient: 'NrrdPatient',
        id: StudyID,
        check_path: bool = True,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__global_id = f'{patient} - {id}'
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
    def patient(self) -> 'NrrdPatient':
        return self.__patient

    def has_data(
        self,
        id: str,
        modality: Modality) -> bool:
        return id in self.list_data(modality)

    def has_landmark(self, *args, **kwargs) -> bool:
        return self.default_landmarks.has_landmark(*args, **kwargs)

    def has_region(self, *args, **kwargs) -> bool:
        return self.default_regions.has_region(*args, **kwargs)

    def landmark_data(self, *args, **kwargs) -> Landmarks:
        return self.default_landmarks.data(*args, **kwargs)

    def list_landmarks(self, *args, **kwargs) -> List[PatientLandmark]:
        return self.default_landmarks.list_landmarks(*args, **kwargs)

    def list_regions(self, *args, **kwargs) -> List[PatientRegion]:
        return self.default_regions.list_regions(*args, **kwargs)

    def list_data(
        self,
        modality: Modality) -> List[str]:
        if modality == Modality.CT:
            data_ids = list(sorted(os.listdir(os.path.join(self.__path, 'ct'))))
            data_ids = [s.replace('.nrrd', '') for s in data_ids]
        elif modality == Modality.LANDMARKS:
            data_ids = list(sorted(os.listdir(os.path.join(self.__path, 'landmarks'))))
            data_ids = [s.replace('.csv', '') for s in data_ids]
        elif modality == Modality.REGIONS:
            data_ids = list(sorted(os.listdir(os.path.join(self.__path, 'regions'))))
        elif modality == Modality.DOSE:
            pass
    
        return data_ids

    def region_data(self, *args, **kwargs) -> Dict[PatientRegion, np.ndarray]:
        return self.default_regions.data(*args, **kwargs)

    def region_path(self, *args, **kwargs) -> str:
        return self.default_regions.region_path(*args, **kwargs)

    def data(
        self,
        id: str,
        modality: Modality) -> NrrdData:
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
    