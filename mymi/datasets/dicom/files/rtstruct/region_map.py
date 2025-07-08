import numpy as np
import os
import pandas as pd
import re
from typing import *

from mymi.regions import is_region
from mymi.typing import *

class RegionMap:
    def __init__(
        self,
        data: Dict[str, str]) -> None:
        self.__data = data

    @staticmethod
    def load(filepath: str) -> Optional['RegionMap']:
        print('loading')
        return RegionMap(load_yaml(filepath)) if os.path.exists(filepath) else None

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    def to_internal(
        self,
        region_id: RegionID,
        pat_id: Optional[PatientID] = None,
        study_id: Optional[StudyID] = None) -> Tuple[RegionID, float]:
        # Check region map for matches.
        for unmapped_region_id, mapped_region_id in self.__data.items():
            if re.match(unmapped_region_id, region_id):
                return mapped_region_id 
            
        return region_id
