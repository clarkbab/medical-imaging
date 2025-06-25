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
    def ct_affine(self) -> Optional[np.ndarray]:
        return self.default_ct.affine if self.default_ct is not None else None

    @property
    def ct_data(self) -> Optional[CtImage]:
        return self.default_ct.data if self.default_ct is not None else None

    def ct_extent(
        self,
        **kwargs) -> Optional[Point3D]:
        return self.default_ct.extent(**kwargs) if self.default_ct is not None else None

    @property
    def ct_fov(self) -> Optional[ImageFOV3D]:
        return self.default_ct.fov if self.default_ct is not None else None

    @property
    def ct_offset(self) -> Optional[Point3D]:
        return self.default_ct.offset if self.default_ct is not None else None

    @property
    def ct_path(self) -> Optional[str]:
        return self.default_ct.path if self.default_ct is not None else None

    @property
    def ct_size(self) -> Optional[Size3D]:
        return self.default_ct.size if self.default_ct is not None else None

    @property
    def ct_spacing(self) -> Optional[Spacing3D]:
        return self.default_ct.spacing if self.default_ct is not None else None

    @property
    def default_ct(self) -> Optional[CtNiftiData]:
        data_ids = self.list_data('CT')
        if len(data_ids) == 0:
            return None
        else:
            return CtNiftiData(self, data_ids[-1])

    @property
    def default_dose(self) -> Optional[CtNiftiData]:
        data_ids = self.list_data('DOSE')
        if len(data_ids) == 0:
            return None
        else:
            return DoseNiftiData(self, data_ids[-1])

    @property
    def default_landmarks(self) -> Optional[LandmarkNiftiData]:
        data_ids = self.list_data('LANDMARKS')
        if len(data_ids) == 0:
            return None
        else:
            return LandmarkNiftiData(self, data_ids[-1])

    @property
    def default_regions(self) -> Optional[RegionNiftiData]:
        data_ids = self.list_data('REGIONS')
        if len(data_ids) == 0:
            return None
        else:
            return RegionNiftiData(self, data_ids[-1], region_map=self.__region_map)

    @property
    def dose_data(self) -> Optional[DoseImage]:
        return self.default_dose.data if self.default_dose is not None else None

    @property
    def dose_path(self) -> Optional[str]:
        return self.default_dose.path if self.default_dose is not None else None

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
        modality: NiftiModality) -> bool:
        return id in self.list_data(modality)

    def has_landmarks(self, *args, **kwargs) -> bool:
        lms = self.default_landmarks
        if lms is None:
            return False
        else:
            return lms.has_landmarks(*args, **kwargs)

    def has_regions(self, *args, **kwargs) -> bool:
        def_regions = self.default_regions 
        if def_regions is None:
            return False
        return def_regions.has_regions(*args, **kwargs)

    def landmark_data(self, *args, **kwargs) -> Optional[LandmarkData]:
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
        modality: NiftiModality) -> List[SeriesID]:
        if modality == 'CT':
            filepath = os.path.join(self.__path, 'ct')
            if os.path.exists(filepath):
                data_ids = list(sorted(os.listdir(filepath)))
                data_ids = [s.replace('.nii.gz', '') for s in data_ids]
            else:
                data_ids = []
        elif modality == 'DOSE':
            filepath = os.path.join(self.__path, 'dose')
            if os.path.exists(filepath):
                data_ids = list(sorted(os.listdir(filepath)))
                data_ids = [s.replace('.nii.gz', '') for s in data_ids]
            else:
                data_ids = []
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
        modality: Optional[NiftiModality] = None) -> NiftiData:
        if modality is None:
            modality = self.data_modality(id)

        if modality == 'CT':
            data = CtNiftiData(self, id)
        elif modality == 'DOSE':
            pass
        elif modality == 'LANDMARKS':
            data = LandmarkNiftiData(self, id)
        elif modality == 'MR':
            data = MrNiftiData(self, id)
        elif modality == 'REGIONS':
            data = RegionNiftiData(self, id, region_map=self.__region_map)
        else:
            raise ValueError(f"Modality '{modality}' not supported.")

        return data

    def data_modality(
        self,
        id: SeriesID) -> NiftiModality:
        modalities = get_args(NiftiModality)
        for m in modalities:
            data_ids = self.list_data(m)
            if id in data_ids:
                return m
        raise ValueError(f"Data with ID '{id}' not found in study '{self.__global_id}'.")

    def __str__(self) -> str:
        return self.__global_id
    