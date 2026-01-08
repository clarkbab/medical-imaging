import numpy as np
import os
import pandas as pd
import re
from typing import *

from mymi.regions import is_region
from mymi.typing import *
from mymi.utils import *

class RegionMap:
    def __init__(
        self,
        data: Dict[RegionID, RegionID]) -> None:
        self.__data = data

    @property
    def data(self) -> Dict[RegionID, RegionID]:
        return self.__data

    def inv_map_region(
        self,
        region: RegionID,
        disk_regions: RegionIDs = []) -> Union[RegionID, List[RegionID]]:
        disk_regions = arg_to_list(disk_regions, RegionID)
        # Takes a single mapped region ID and returns (potentially) multiple unmapped region IDs.
        # E.g. 'Brainstem' -> 'BrainStem'
        # 'Chestwall' -> ['Chestwall_L', 'Chestwall_R'].
        # Used for getting disk regions from mapped regions.
        inv_mapped_ids = []

        # Check literal matches.
        literals = self.__data['literals'] if 'literals' in self.__data else self.__data if 'regexes' not in self.__data else None
        if literals is not None:
            for k, v in literals.items():
                if v == region:
                    inv_mapped_ids.append(k)

        # What if the mapped region comes from a regex? We'll need the patient ID to get the disk region ID.
        # E.g. '$Chestwall_(L|R)$' -> 'Chestwall'.

        # Check regex matches.
        regexes = self.__data['regexes'] if 'regexes' in self.__data else None
        if regexes is not None:
            for k, v in regexes.items():
                if v == region:
                    if len(disk_regions) > 0:
                        for r in disk_regions:
                            if re.match(k, r, flags=re.IGNORECASE):
                                inv_mapped_ids.append(r)
                    else:
                        raise ValueError(f"Cannot perform inverse mapping for regex '{k}' without a list of 'disk_regions'.")

        return inv_mapped_ids if len(inv_mapped_ids) > 1 else inv_mapped_ids[0] if len(inv_mapped_ids) == 1 else region

    def map_region(
        self,
        region: RegionID) -> RegionID:
        # Check literal matches.
        literals = self.__data['literals'] if 'literals' in self.__data else self.__data if 'regexes' not in self.__data else None
        if literals is not None:
            for k, v in literals.items():
                if k == region:
                    return v

        # Check regex matches.
        regexes = self.__data['regexes'] if 'regexes' in self.__data else None
        if regexes is not None:
            for k, v in regexes.items():
                if re.match(k, region, flags=re.IGNORECASE):
                    return v

        return region
