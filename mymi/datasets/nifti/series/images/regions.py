import numpy as np
import os
from typing import *

from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ....region_map import RegionMap
from .images import NiftiImageSeries

class RegionsImageSeries(NiftiImageSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID,
        path: DirPath,
        region_map: Optional[RegionMap] = None) -> None:
        self.__dataset_id = dataset_id
        self._global_id = f'NIFTI:{dataset_id}:{pat_id}{study_id}:{id}'
        self.__id = id
        self.__path = path
        self.__pat_id = pat_id
        self.__region_map = region_map
        self.__study_id = study_id

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
            # If multiple regions have been mapped to the same ID, then get the union of these regions.
            filepaths = self.filepaths(r)
            ds = []
            for f in filepaths:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    d, _, _ = load_nifti(f)
                elif f.endswith('.nrrd'):
                    d, _, _ = load_nrrd(f)
                else:
                    raise ValueError(f'Unsupported file format: {f}')
                ds.append(d)
            rd[r] = np.sum(ds, axis=0).clip(0, 1).astype(bool)

        return rd

    def filepaths(
        self,
        region_id: RegionID) -> List[FilePath]:
        if not self.has_regions(region_id):
            raise ValueError(f'Region {region_id} not found in series {self.id}.')
        # Region mapping is multi-to-one, so we could get multiple files on disk for the same region.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        disk_ids = self.__region_map.inv_map_region(region_id) if self.__region_map is not None else region_id
        disk_ids = arg_to_list(disk_ids, RegionID)
        # Check all possible file extensions.
        filepaths = [os.path.join(self.__path, f'{i}{e}') for i in disk_ids for e in image_extensions if os.path.exists(os.path.join(self.__path, f'{i}{e}'))]
        if len(filepaths) == 0:
            raise ValueError(f'No region filespaths found for region {region_id} in series {self}.')
        return filepaths

    def has_regions(
        self,
        region_ids: RegionIDs = 'all',
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_regions(**kwargs)
        region_ids = arg_to_list(region_ids, RegionID, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(region_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(region_ids)

    def list_regions(
        self,
        region_ids: Regions = 'all') -> List[Region]:
        # Load regions from filenames.
        image_extensions = ['.nii', '.nii.gz', '.nrrd']
        ids = os.listdir(self.__path)
        ids = [i.replace(e, '') for i in ids for e in image_extensions if i.endswith(e)]

        # Apply region mapping.
        if self.__region_map is not None:
            ids = [self.__region_map.map_region(i) if self.__region_map is not None else i for i in ids]

        # Filter on 'only'.
        if region_ids != 'all':
            region_ids = regions_to_list(region_ids)
            ids = [r for r in ids if r in region_ids]

        # Sort regions.
        ids = list(sorted(ids))

        return ids
    
# Add properties.
props = ['id', 'path']
for p in props:
    setattr(RegionsImageSeries, p, property(lambda self, p=p: getattr(self, f'_{RegionsImageSeries.__name__}__{p}')))
