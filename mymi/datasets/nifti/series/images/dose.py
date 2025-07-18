import numpy as np
import os

from mymi.geometry import get_extent
from mymi.typing import *
from mymi.utils import *

from .images import NiftiImage

class DoseImage(NiftiImage):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__global_id = f'{study}:{id}'
        self.__filepath = os.path.join(study.path, 'dose', f'{id}.nii.gz')

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__data, self.__spacing, self.__offset = load_nifti(self.__filepath)
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> DoseData:
        return self.__data

    @ensure_loaded
    def extent(
        self,
        use_patient_coords: bool = True) -> Union[Point3D, Voxel]:
        return get_extent(self.__data, spacing=self.__spacing, offset=self.__offset, use_patient_coords=use_patient_coords)

    @property
    @ensure_loaded
    def fov(
        self,
        **kwargs) -> Union[FOV3D, Size3D]:
        ext_min, ext_max = self.extent(**kwargs)
        fov = tuple(np.array(ext_max) - ext_min)
        return fov

    @property
    @ensure_loaded
    def offset(self) -> Point3D:
        return self.__offset

    @property
    @ensure_loaded
    def size(self) -> Size3D:
        return self.__data.shape

    @property
    @ensure_loaded
    def spacing(self) -> Spacing3D:
        return self.__spacing

# Add properties.
props = ['filepath', 'global_id', 'id']
for p in props:
    setattr(DoseImage, p, property(lambda self, p=p: getattr(self, f'_{DoseImage.__name__}__{p}')))
