import numpy as np
import os
import pandas as pd
import re
from typing import Optional, Tuple

from mymi.regions import is_region
from mymi.types import PatientID

class RegionMap:
    def __init__(
        self,
        data: pd.DataFrame):
        self.__data = data

    @staticmethod
    def load(filepath: str) -> Optional['RegionMap']:
        if os.path.exists(filepath):
            map_df = pd.read_csv(filepath, dtype={ 'patient-id': str })

            # # Check that internal region names are entered correctly.
            # for region in map_df.internal:
            #     if not is_region(region):
            #         raise ValueError(f"Error in region map. '{region}' is not an internal region.")
            
            return RegionMap(map_df)
        else:
            return None

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    def to_internal(
        self,
        region: str,
        pat_id: Optional[PatientID] = None) -> Tuple[str, int]:

        # Iterate over map rows.
        match = None
        priority = -np.inf
        for _, row in self.__data.iterrows():
            if 'patient-id' in row:
                # Skip if this map row is for a different patient.
                map_pat_id = row['patient-id']
                if isinstance(map_pat_id, str) and str(pat_id) != map_pat_id:
                    continue

            args = []
            # Add case sensitivity to regexp match args.
            if 'case' in row:
                case = row['case']
                if not np.isnan(case) and not case:
                    args += [re.IGNORECASE]
            else:
                args += [re.IGNORECASE]
                
            # Perform match.
            if re.match(row['dataset'], region, *args):
                if 'priority' in row and not np.isnan(row['priority']):
                    if row['priority'] > priority:
                        match = row['internal']
                        priority = row['priority']
                else:
                    match = row['internal']

        if match is None:
            match = region
        
        return match, priority
