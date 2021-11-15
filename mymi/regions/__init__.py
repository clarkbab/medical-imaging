from typing import List

from mymi import types

from .colours import to_255, RegionColours
from .patches import get_patch_size
from .regions import Regions

def is_region(name: str) -> bool:
    """
    returns: if the name is an internal region.
    args:
        name: the name of the region.
    """
    # Get region names.
    names = [r.name for r in Regions]
    return name in names

def to_list(
    regions: types.PatientRegions,
    all_regions: List[str]) -> List[str]:
    """
    returns: a list of regions names.
    args:
        regions: the regions argument.
        all_regions: all possible regions.
    """
    if type(regions) == str:
        if regions == 'all':
            return all_regions
        else:
            return [regions]
    else:
        return regions

