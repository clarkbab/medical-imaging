import numpy as np
import scipy
from typing import *

from mymi.typing import *

def get_centre_of_mass(
    a: Union[ImageData2D, ImageData3D],
    offset: Optional[Union[Point2D, Point3D]] = None,
    spacing: Optional[Union[Spacing2D, Spacing3D]] = None, 
    use_patient_coords: bool = True) -> Union[Point2D, Point3D]:
    com = scipy.ndimage.center_of_mass(a)
    if use_patient_coords:
        com = tuple(np.array(com) * spacing + offset)
    else:
        com = tuple([int(np.round(c)) for c in com])
    return com
