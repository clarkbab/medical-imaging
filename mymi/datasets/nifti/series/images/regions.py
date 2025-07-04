import numpy as np
import os
from typing import *

from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from .images import NiftiImage

class RegionsImage(NiftiImage):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID,
        region_map: Optional[Dict[str, str]] = None) -> None:
        self.__id = id
        self.__global_id = f'{study}:{id}'
        self.__path = os.path.join(study.path, 'regions', id)
        self.__region_map = region_map
        self.__inv_region_map = {v: k for k, v in self.__region_map.items()} if self.__region_map is not None else None

    def data(
        self,
        region_ids: Regions = 'all',
        regions_ignore_missing: bool = True,
        **kwargs) -> RegionsData:
        region_ids = regions_to_list(region_ids, literals={ 'all': self.list_regions })

        rd = {}
        for r in region_ids:
            if not self.has_regions(r):
                if regions_ignore_missing:
                    continue
                else:
                    raise ValueError(f'Region {r} not found in image {self.id}.')

            # Load region from disk.
            d, _, _ = load_nifti(self.filepath(r))
            rd[r] = d.astype(bool)

        return rd

    def filepath(
        self,
        region_id: RegionID) -> str:
        if not self.has_regions(region_id):
            raise ValueError(f'Region {region_id} not found in series {self.id}.')
        # Account for potential region mapping.
        disk_region_id = self.__inv_region_map[region_id] if self.__inv_region_map is not None and region_id in self.__inv_region_map else region_id
        return os.path.join(self.__path, f'{disk_region_id}.nii.gz')
    
    def has_regions(
        self,
        region_ids: RegionIDs,
        any: bool = False) -> bool:
        real_ids = self.list_regions(region_ids=region_ids)
        req_ids = arg_to_list(region_ids, RegionID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def list_regions(
        self,
        region_ids: Regions = 'all') -> List[Region]:
        # Load regions from filenames.
        ids = os.listdir(self.__path)
        ids = [r.replace('.nii.gz', '') for r in ids]

        # Apply region mapping.
        if self.__region_map is not None:
            ids = [self.__region_map[r] if r in self.__region_map else r for r in ids]

        # Filter on 'only'.
        if region_ids != 'all':
            region_ids = regions_to_list(region_ids)
            ids = [r for r in ids if r in region_ids]

        # Sort regions.
        ids = list(sorted(ids))

        return ids
    
# Add properties.
props = ['global_id', 'id']
for p in props:
    setattr(RegionsImage, p, property(lambda self, p=p: getattr(self, f'_{RegionsImage.__name__}__{p}')))
