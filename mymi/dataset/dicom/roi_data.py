from dataclasses import dataclass
import numpy as np
import pydicom as dcm
from typing import Optional

from mymi import types

@dataclass
class ROIData:
    colour: types.Colour
    data: np.ndarray
    name: str
    number: Optional[int] = None
