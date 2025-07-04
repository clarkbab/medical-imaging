import pandas as pd
from typing import *

from mymi.typing import *

from ..files import RegionMap, RtStructFile
from .ct import CtSeries
from .series import DicomSeries

class RtStructSeries(DicomSeries):
    def __init__(
        self,
        study: 'DicomStudy',
        id: SeriesID,
        region_map: Optional[RegionMap] = None) -> None:
        self.__global_id = f"{study}:{id}"
        self.__id = id
        self.__region_map = region_map
        self.__study = study

        # Get index.
        index = self.__study.index
        index = index[(index.modality == 'rtstruct') & (index['series-id'] == self.__id)].copy()
        if len(index) == 0:
            raise ValueError(f"No RTSTRUCT series with ID '{id}' found in study '{study}'.")
        self.__index = index

        # Get policies.
        self.__index_policy = self.__study.index_policy['rtstruct']
        self.__region_policy = self.__study.region_policy

    @property
    def default_file(self) -> RtStructFile:
        # Choose most recent RTSTRUCT.
        rtstruct_ids = self.list_files()
        return self.file(rtstruct_ids[-1])

    @property
    def description(self) -> str:
        return self.__global_id

    def file(
        self,
        id: DicomSOPInstanceUID) -> RtStructFile:
        return RtStructFile(self, id, region_map=self.__region_map)

    def landmarks_data(self, *args, **kwargs):
        return self.default_file.landmarks_data(*args, **kwargs)

    def list_files(self) -> List[DicomSOPInstanceUID]:
        return list(sorted(self.__index.index))

    @property
    def modality(self) -> DicomModality:
        return 'rtstruct'

    @property
    def ref_ct(self) -> CtSeries:
        return self.default_file.ref_ct

    def regions_data(self, *args, **kwargs):
        return self.default_file.regions_data(*args, **kwargs)

# Add properties.
props = ['global_id', 'id', 'index', 'index_policy', 'region_map', 'region_policy', 'study']
for p in props:
    setattr(RtStructSeries, p, property(lambda self, p=p: getattr(self, f'_{RtStructSeries.__name__}__{p}')))

# Add property shortcuts from 'default_file'.
props = ['filepath', 'ref_ct']
for p in props:
    setattr(RtStructSeries, p, property(lambda self, p=p: getattr(self.default_file, p) if self.default_file is not None else None))

# Add landmark/region method shortcuts from 'default_file'.
mods = ['landmark', 'region']
for m in mods:
    setattr(RtStructSeries, f'list_{m}s', lambda self, *args, m=m, **kwargs: getattr(self.default_file, f'list_{m}s')(*args, **kwargs) if self.default_file is not None else [])
    setattr(RtStructSeries, f'has_{m}s', lambda self, *args, m=m, **kwargs: getattr(self.default_file, f'has_{m}s')(*args, **kwargs) if self.default_file is not None else False)
