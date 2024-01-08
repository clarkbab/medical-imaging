from typing import List

from mymi import types

from .colours import to_255, RegionColours
from .dose_constraints import get_dose_constraint
from .limits import RegionLimits, truncate_spine
from .list import RegionList, region_to_list
from .patch_sizes import get_region_patch_size
from .regions import RegionNames
from .tolerances import get_region_tolerance, RegionTolerances

def is_region(name: str) -> bool:
    return name in RegionNames
