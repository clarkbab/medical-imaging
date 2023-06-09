import collections
import pandas as pd
import pydicom as dcm
from typing import Dict, List, Optional, OrderedDict, Tuple, Union

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
        return_unmapped: bool = False,
        use_mapping: bool = True) -> Union[List[PatientRegion], Tuple[List[PatientRegion], List[PatientRegion]]]:
        # Get unmapped region names.
        rtstruct = self.get_rtstruct()
        unmapped_regions = RTSTRUCTConverter.get_roi_names(rtstruct)

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        unmapped_regions = list(filter(lambda r: RTSTRUCTConverter.has_roi_data(rtstruct, r), unmapped_regions))

        # Map regions using 'region-map.csv'.
        if self.__region_map is None:
            use_mapping = False
        if use_mapping:
            pat_id = self.__series.study.patient.id
            # Store as ('unmapped region', 'mapped region', 'priority').
            mapped_regions = []
            for unmapped_region in unmapped_regions:
                mapped_region, priority = self.__region_map.to_internal(unmapped_region, pat_id=pat_id)
                # Don't map regions that would map to an existing region name.
                if mapped_region != unmapped_region and mapped_region in unmapped_regions:
                    logging.warning(f"Mapped region '{mapped_region}' (mapped from '{unmapped_region}') already found in unmapped regions for '{self}'. Skipping.")
                    mapped_regions.append((unmapped_region, mapped_region, priority))

                # Don't map regions that are already present in 'new_regions'.
                elif mapped_region in mapped_regions:
                    raise ValueError(f"Mapped region '{mapped_region}' (mapped from '{unmapped_region}') already found in mapped regions for '{self}'. Set 'priority' in region map.")

                # Map region.
                else:
                    mapped_regions.append((unmapped_region, mapped_region, priority))

            # If multiple unmapped regions map to the same region, then choose to map the one with
            # higher priorty.
            for i in range(len(mapped_regions)):
                unmapped_region, mapped_region, priority = mapped_regions[i]
                for _, mr, p in mapped_regions:
                    # If another mapped region exists with a higher priority, then set this region
                    # back to its unmapped form.
                    if mr == mapped_region and p > priority:
                        mapped_regions[i] = (unmapped_region, unmapped_region, priority)

            # Remove priority.
            mapped_regions = [r[:-1] for r in mapped_regions]

        # Filter on 'only'. If region mapping is used (i.e. mapped_regions != None),
        # this will try to match mapped names, otherwise it will map unmapped names.
        if only is not None:
            only = arg_to_list(only, str)

            if use_mapping:
                mapped_regions = [r for r in mapped_regions if r[1] in only]
            else:
                unmapped_regions = [r for r in unmapped_regions if r in only]

        # Check for multiple regions.
        if not self.__region_policy['duplicates']['allow']:
            if use_mapping:
                # Only check for duplicates on mapped regions.
                names = [r[1] for r in mapped_regions]
            else:
                names = unmapped_regions

            # Get duplicated regions.
            dup_regions = [r for r in names if names.count(r) > 1]

            if len(dup_regions) > 0:
                if use_mapping and self.__region_map is not None:
                    raise ValueError(f"Duplicate regions found for RTSTRUCT '{self}', perhaps a 'region-map.csv' issue? Duplicated regions: '{dup_regions}'")
                else:
                    raise ValueError(f"Duplicate regions found for RTSTRUCT '{self}'. Duplicated regions: '{dup_regions}'")

        # Sort regions.
        if use_mapping:
            mapped_regions = list(sorted(mapped_regions, key=lambda r: r[1]))
        else:
            unmapped_regions = list(sorted(unmapped_regions))

        # Choose return type when using mapping.
        if use_mapping:
            if return_unmapped:
                return mapped_regions
            else:
                mapped_regions = [r[1] for r in mapped_regions]
                return mapped_regions
        else:
            return unmapped_regions

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

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        pat_regions = self.list_regions(return_unmapped=True, use_mapping=use_mapping)

        # Filter on requested regions.
        if regions is not None:
            if use_mapping:
                pat_regions = [r for r in pat_regions if r[1] in regions]
            else:
                pat_regions = [r for r in pat_regions if r in regions]

        # Get reference CTs.
        cts = self.ref_ct.get_cts()

        # Load RTSTRUCT dicom.
        rtstruct = self.get_rtstruct()

        # Add ROI data.
        results = {}
        if use_mapping:
            # Load region using unmapped name, store using mapped name.
            for unmapped_region, mapped_region in pat_regions:
                data = RTSTRUCTConverter.get_roi_data(rtstruct, unmapped_region, cts)
                results[mapped_region] = data
        else:
            # Load and store region using unmapped name.
            for region in pat_regions:
                data = RTSTRUCTConverter.get_roi_data(rtstruct, region, cts)
                results[region] = data

        # Create ordered dict.
        results = collections.OrderedDict((n, results[n]) for n in sorted(results.keys())) 

        return results

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
