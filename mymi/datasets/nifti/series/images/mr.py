import numpy as np
import os

from mymi.typing import *
from mymi.utils import *

from .images import NiftiImage

class MrImage(NiftiImage):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__global_id = f'{study}:{id}'
        self.__filepath = os.path.join(study.path, 'mr', f'{id}.nii.gz')

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__data'):
                self.__data, self.__spacing, self.__offset = load_nifti(self.__filepath)
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    @ensure_loaded
    def data(self) -> MrData:
        return self.__data

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
    def spacing(self) -> np.ndarray:
        return self.__spacing

# Add properties.
props = ['filepath', 'global_id', 'id']
for p in props:
    setattr(MrImage, p, property(lambda self, p=p: getattr(self, f'_{MrImage.__name__}__{p}')))
