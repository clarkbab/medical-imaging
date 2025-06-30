import numpy as np
import os
from typing import Dict, List, Optional

from mymi.regions import regions_to_list
from mymi.typing import Region, Regions, SeriesID
from mymi.utils import load_nrrd

from .data import NrrdData

class RegionImage(NrrdData):
    def __init__(
        self,
        study: 'NrrdStudy',
        id: SeriesID,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__id = id
        self.__path = os.path.join(study.path, 'regions', id)
        self.__region_map = region_map
        self.__inv_region_map = {v: k for k, v in self.__region_map.items()} if self.__region_map is not None else None

    def data(
        self,
        regions: Regions = 'all',
        regions_ignore_missing: bool = False,
        **kwargs) -> Dict[str, np.ndarray]:
        regions = regions_to_list(regions, literals={ 'all': self.list_regions })

        rd = {}
        for r in regions:
            if not self.has_regions(r):
                if regions_ignore_missing:
                    continue
                else:
                    raise ValueError(f"Requested region '{r}' not present for RTSTRUCT {self}")

            # Load region from disk.
            disk_region = self.__inv_region_map[r] if self.__inv_region_map is not None and r in self.__inv_region_map else r
            filepath = os.path.join(self.__path, f'{disk_region}.nrrd')
            d, _, _ = load_nrrd(filepath)
            rd[r] = d.astype(bool)

        return rd
    
    # Returns 'True' if has at least one of the passed 'regions'.
    def has_regions(
        self,
        regions: Regions) -> bool:
        regions = regions_to_list(regions, literals={ 'all': self.list_regions })
        pat_regions = self.list_regions()
        if len(np.intersect1d(regions, pat_regions)) != 0:
            return True
        else:
            return False

    def list_regions(
        self,
        # Only the regions in 'regions' should be returned.
        # Saves us from performing filtering code elsewhere many times.
        regions: Optional[Regions] = None) -> List[Region]:
        regions = regions_to_list(regions)

        # Load regions from filenames.
        rs = os.listdir(self.__path)
        rs = [r.replace('.nrrd', '') for r in rs]

        # Apply region mapping.
        if self.__region_map is not None:
            rs = [self.__region_map[r] if r in self.__region_map else r for r in rs]

        # Filter on 'only'.
        if regions is not None:
            rs = [r for r in rs if r in regions]

        # Sort regions.
        rs = list(sorted(rs))

        return rs

    def region_path(
        self,
        region: Region) -> str:
        return os.path.join(self.__path, f'{region}.nrrd')
    