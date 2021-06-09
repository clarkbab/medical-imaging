from .colours import to_255, RegionColours
from .regions import Regions

def is_region(name: str) -> bool:
    """
    returns: if the name is an internal region.
    args:
        name: the name of the region.
    """
    # Get region names.
    region_names = [r.name for r in Regions]
    return name in region_names
