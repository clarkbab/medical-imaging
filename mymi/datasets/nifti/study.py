import numpy as np
import os
from typing import *

from mymi.typing import *

from .images import *

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

    def ct_extent(
        self,
        **kwargs) -> Optional[Point3D]:
        return self.default_ct.extent(**kwargs) if self.default_ct is not None else None

    @property
    def default_ct(self) -> Optional[CtNiftiImage]:
        ct_ids = self.list_images('ct')
        if len(ct_ids) == 0:
            return None
        else:
            return CtNiftiImage(self, ct_ids[-1])

    @property
    def default_dose(self) -> Optional[CtNiftiImage]:
        dose_ids = self.list_images('dose')
        if len(dose_ids) == 0:
            return None
        else:
            return DoseNiftiImage(self, dose_ids[-1])

    @property
    def default_landmarks(self) -> Optional[LandmarksNifti]:
        landmark_ids = self.list_landmarks()
        if len(landmark_ids) == 0:
            return None
        else:
            return LandmarksNifti(self, landmark_ids[-1])

    @property
    def default_regions(self) -> Optional[RegionNiftiImage]:
        regions_ids = self.list_images('regions')
        if len(regions_ids) == 0:
            return None
        else:
            return RegionNiftiImage(self, regions_ids[-1], region_map=self.__region_map)

    @property
    def id(self) -> str:
        return self.__id

    @property
    def path(self) -> str:
        return self.__path

    @property
    def patient(self) -> 'NiftiPatient':
        return self.__patient

    @property
    def has_dose(self) -> bool:
        return self.default_dose is not None

    def has_image(
        self,
        id: str,
        modality: NiftiModality) -> bool:
        return id in self.list_images(modality)

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

    def list_images(
        self,
        modality: NiftiModality) -> List[SeriesID]:
        if modality == 'ct':
            filepath = os.path.join(self.__path, 'ct')
            if os.path.exists(filepath):
                image_ids = list(sorted(os.listdir(filepath)))
                image_ids = [s.replace('.nii.gz', '') for s in image_ids]
            else:
                image_ids = []
        elif modality == 'dose':
            filepath = os.path.join(self.__path, 'dose')
            if os.path.exists(filepath):
                image_ids = list(sorted(os.listdir(filepath)))
                image_ids = [s.replace('.nii.gz', '') for s in image_ids]
            else:
                image_ids = []
        # elif modality == 'landmarks':
        #     filepath  = os.path.join(self.__path, 'landmarks')
        #     if os.path.exists(filepath):
        #         image_ids = list(sorted(os.listdir(filepath)))
        #         image_ids = [s.replace('.csv', '') for s in image_ids]
        #     else:
        #         image_ids = []
        elif modality == 'mr':
            filepath = os.path.join(self.__path, 'mr')
            if os.path.exists(filepath):
                image_ids = list(sorted(os.listdir(filepath)))
                image_ids = [s.replace('.nii.gz', '') for s in image_ids]
            else:
                image_ids = []
        elif modality == 'regions':
            filepath = os.path.join(self.__path, 'regions')
            if os.path.exists(filepath):
                image_ids = list(sorted(os.listdir(filepath)))
            else:
                image_ids = []
        else:
            raise ValueError(f"Modality '{modality}' not supported.")
    
        return image_ids

    def list_landmarks(self) -> List[SeriesID]:
        filepath  = os.path.join(self.__path, 'landmarks')
        if os.path.exists(filepath):
            landmark_ids = list(sorted(os.listdir(filepath)))
            landmark_ids = [i.replace('.csv', '') for i in landmark_ids]
        else:
            landmark_ids = []
        return landmark_ids

    def region_images(self, *args, **kwargs) -> Optional[RegionImage]:
        def_regions = self.default_regions 
        if def_regions is None:
            return None
        return def_regions.data(*args, **kwargs)

    def region_path(self, *args, **kwargs) -> Optional[str]:
        def_regions = self.default_regions 
        if def_regions is None:
            return None
        return def_regions.region_path(*args, **kwargs)

    def image(
        self,
        id: str,
        modality: Optional[NiftiModality] = None) -> NiftiImage:
        if modality is None:
            modality = self.image_modality(id)

        if modality == 'ct':
            image = CtNiftiImage(self, id)
        elif modality == 'dose':
            pass
        # elif modality == 'landmarks':
        #     image = LandmarksNifti(self, id)
        elif modality == 'mr':
            image = MrNiftiImage(self, id)
        elif modality == 'regions':
            image = RegionNiftiImage(self, id, region_map=self.__region_map)
        else:
            raise ValueError(f"Modality '{modality}' not supported.")

        return image

    def image_modality(
        self,
        id: SeriesID) -> NiftiModality:
        modalities = get_args(NiftiModality)
        for m in modalities:
            image_ids = self.list_images(m)
            if id in image_ids:
                return m
        raise ValueError(f"Image with ID '{id}' not found in study '{self.__global_id}'.")

    def __str__(self) -> str:
        return self.__global_id
    
# Add 'has_{mod}' properties.
mods = ['ct', 'dose', 'landmarks', 'mr', 'regions']
for m in mods:
    setattr(NiftiStudy, f'has_{m}', property(lambda self, m=m: getattr(self, f'default_{m}') is not None))

# Add image property shortcuts.
mods = ['ct', 'mr', 'dose']
props = ['data', 'fov', 'offset', 'size', 'spacing']
for m in mods:
    for p in props:
        setattr(NiftiStudy, f'{m}_{p}', property(lambda self, m=m, p=p: getattr(getattr(self, f'default_{m}'), p) if getattr(self, f'default_{m}') is not None else None))
