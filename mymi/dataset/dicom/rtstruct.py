import collections
import pandas as pd
import pydicom as dcm
from typing import Dict, List, Optional, OrderedDict

from mymi import logging
from mymi.types import PatientRegion, PatientRegions
from mymi.utils import arg_to_list

from .ct_series import CTSeries
from .dicom_file import DICOMFile, SOPInstanceUID
from .dicom_series import DICOMModality
from .region_map import RegionMap
from .rtstruct_converter import RTSTRUCTConverter

class RTSTRUCT(DICOMFile):
    def __init__(
        self,
        series: 'RTSTRUCTSeries',
        id: SOPInstanceUID,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None):
        self.__global_id = f"{series} - {id}"
        self.__id = id
        self.__ref_ct = None        # Lazy-loaded.
        self.__region_dups = region_dups
        self.__region_map = region_map
        self.__series = series

        # Get index.
        index = self.__series.index
        self.__index = index.loc[[self.__id]]
        self.__verify_index()
        self.__path = self.__index.iloc[0]['filepath']

        # Get policies.
        self.__index_policy = self.__series.index_policy
        self.__region_policy = self.__series.region_policy

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def id(self) -> SOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def index_policy(self) -> pd.DataFrame:
        return self.__index_policy

    @property
    def path(self) -> str:
        return self.__path

    @property
    def ref_ct(self) -> str:
        if self.__ref_ct is None:
            self.__load_ref_ct()
        return self.__ref_ct

    @property
    def region_policy(self) -> pd.DataFrame:
        return self.__region_policy

    def get_rtstruct(self) -> dcm.dataset.FileDataset:
        return dcm.read_file(self.__path)

    def get_region_info(
        self,
        use_mapping: bool = True) -> Dict[int, Dict[str, str]]:
        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Get region IDs.
        roi_info = RTSTRUCTConverter.get_roi_info(rtstruct)

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        roi_info = dict(filter(lambda i: RTSTRUCTConverter.has_roi_data(rtstruct, i[1]['name']), roi_info.items()))

        # Map to internal names.
        if use_mapping and self.__region_map:
            pat_id = self.__series.study.patient.id
            def map_name(info):
                info['name'] = self.__region_map.to_internal(info['name'], pat_id=pat_id)
                return info
            roi_info = dict((id, map_name(info)) for id, info in roi_info.items())

        return roi_info

    def has_region(
        self,
        region: PatientRegion,
        use_mapping: bool = True) -> bool:
        return region in self.list_regions(only=region, use_mapping=use_mapping)

    def list_regions(
        self,
        only: Optional[PatientRegions] = None,
        use_mapping: bool = True) -> List[PatientRegion]:
        # Get region names.
        rtstruct = self.get_rtstruct()
        regions = list(sorted(RTSTRUCTConverter.get_roi_names(rtstruct)))

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        regions = list(filter(lambda r: RTSTRUCTConverter.has_roi_data(rtstruct, r), regions))

        # Map to internal regions.
        if use_mapping and self.__region_map is not None:
            pat_id = self.__series.study.patient.id
            new_regions = []
            for region in regions:
                mapped_region, priority = self.__region_map.to_internal(region, pat_id=pat_id)
                # Don't map regions that would map to an existing region name.
                if mapped_region != region and mapped_region in regions:
                    logging.warning(f"Mapped region '{mapped_region}' (mapped from '{region}') already found in unmapped regions for '{self}'. Skipping.")
                    new_regions.append((region, priority))
                # Don't map regions that are already present in 'new_regions'.
                elif mapped_region in new_regions:
                    raise ValueError(f"Mapped region '{mapped_region}' (mapped from '{region}') already found in mapped regions for '{self}'. Set 'priority' in region map.")
                else:
                    new_regions.append((mapped_region, priority))

            # Deduplicate 'new_regions' by priority. I.e. if 'GTVp' (priority=0) and 'GTVp' (priority=1) are both present,
            # then choose 'GTVp' (priority=1) and don't map 'GTVp' (priority=0).
            for i in range(len(new_regions)):
                n_r, n_p = new_regions[i]
                for r, p in new_regions:
                    if r == n_r and p > n_p:
                        new_regions[i] = (regions[i], n_p)
            regions = [r[0] for r in new_regions]

        # Filter on 'only'.
        if only is not None:
            only = arg_to_list(only, str)
            regions = [r for r in regions if r in only]

        # Check for multiple regions.
        if not self.__region_policy['duplicates']['allow']:
            dup_regions = [r for r in regions if regions.count(r) > 1]
            if len(dup_regions) > 0:
                if use_mapping and self.__region_map is not None:
                    raise ValueError(f"Duplicate regions found for RTSTRUCT '{self}', perhaps a 'region-map.csv' issue? Duplicated regions: '{dup_regions}'")
                else:
                    raise ValueError(f"Duplicate regions found for RTSTRUCT '{self}'. Duplicated regions: '{dup_regions}'")

        return regions

    def region_data(
        self,
        region: Optional[PatientRegions] = None,
        use_mapping: bool = True) -> OrderedDict:
        # Check that requested regions exist.
        regions = arg_to_list(region, str)
        if regions is not None:
            for region in regions:
                if not self.has_region(region, use_mapping=use_mapping):
                    raise ValueError(f"Requested region '{region}' not present for RTSTRUCT '{self}'.")

        # Get region names - include unmapped as we need these to load RTSTRUCT regions later.
        unmapped_region_names = self.list_regions(use_mapping=False)
        region_names = self.list_regions(use_mapping=use_mapping)
        region_names = list(zip(region_names, unmapped_region_names))

        # Filter on requested regions.
        if regions is not None:
            region_names = list(filter(lambda r: r[0] in regions, region_names))

        # Get reference CTs.
        cts = self.ref_ct.get_cts()

        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Add ROI data.
        region_dict = {}
        for name, unmapped_name in region_names:
            data = RTSTRUCTConverter.get_roi_data(rtstruct, unmapped_name, cts)
            region_dict[name] = data

        # Create ordered dict.
        ordered_dict = collections.OrderedDict((n, region_dict[n]) for n in sorted(region_dict.keys())) 

        return ordered_dict

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RTSTRUCT '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RTSTRUCTs found in index with SOPInstanceUID '{self.__id}' for series '{self.__series}'.")

    def __load_ref_ct(self) -> None:
        if not self.__index_policy['no-ref-ct']['allow']:
            # Get CT series referenced in RTSTRUCT DICOM.
            rtstruct = self.get_rtstruct()
            ct_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID

        elif self.__index_policy['no-ref-ct']['only'] == 'at-least-one-ct' or self.__index_policy['no-ref-ct']['only'] == 'single-ct':
            # Load first CT series in study.
            ct_id = self.__series.study.list_series(DICOMModality.CT)[-1]

        self.__ref_ct = CTSeries(self.__series.study, ct_id)

    def __str__(self) -> str:
        return self.__global_id
