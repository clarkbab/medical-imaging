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
        region_id: RegionID) -> Union[RegionID, List[RegionID]]:
        # Takes a single mapped region ID and returns (potentially) multiple unmapped region IDs.
        # E.g. 'Brainstem' -> 'BrainStem'
        # 'Chestwall' -> ['Chestwall_L', 'Chestwall_R'].

        inv_mapped_ids = []
        for unmapped_id, mapped_id in self.__data.items():
            if mapped_id == region_id:
                inv_mapped_ids.append(unmapped_id)
        return inv_mapped_ids if len(inv_mapped_ids) > 1 else inv_mapped_ids[0] if len(inv_mapped_ids) == 1 else region_id

    def map_region(
        self,
        region_id: RegionID) -> RegionID:
        # Takes a single region ID and returns a single mapped region ID.
        for unmapped_id, mapped_id in self.__data.items():
            if unmapped_id == region_id:
                return mapped_id
        return region_id
