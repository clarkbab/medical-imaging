import collections
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from typing import *

from mymi.constants import *
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

# Must import from series submodules to avoid circular import.
from ...series.ct import CtSeries
from ..files import DicomFile
from .region_map import RegionMap
from .rtstruct_converter import RtStructConverter

class RtStructFile(DicomFile):
    def __init__(
        self,
        series: 'RtStructSeries',
        id: DicomSOPInstanceUID,
        region_map: Optional[RegionMap] = None):
        self.__global_id = f"{series}:{id}"
        self.__id = id
        self.__region_map = region_map
        self.__series = series

        # Get index.
        self.__index = self.__series.index.loc[self.__id].copy()
        self.__filepath = os.path.join(self.__series.study.patient.dataset.path, self.__index['filepath'])

    def ensure_loaded(fn: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            if not has_private_attr(self, '__ref_ct'):
                self.__load_data()
            return fn(self, *args, **kwargs)
        return wrapper

    @property
    def description(self) -> str:
        return self.__global_id

    @property
    def dicom(self) -> dcm.dataset.FileDataset:
        return dcm.read_file(self.__filepath)

    @property
    def id(self) -> DicomSOPInstanceUID:
        return self.__id

    @property
    def index(self) -> pd.DataFrame:
        return self.__index

    @property
    def filepath(self) -> str:
        return self.__filepath

    @property
    @ensure_loaded
    def ref_ct(self) -> str:
        return self.__ref_ct
    
    @property
    def series(self) -> 'RtStructSeries':
        return self.__series

    def get_region_info(
        self,
        use_mapping: bool = True) -> Dict[int, Dict[str, str]]:
        # Load RTSTRUCT dicom.
        rtstruct_dicom = self.dicom

        # Get region IDs.
        roi_info = RtStructConverter.get_roi_info(rtstruct_dicom)

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        roi_info = dict(filter(lambda i: RtStructConverter.has_roi_data(rtstruct_dicom, i[1]['name']), roi_info.items()))

        # Map to internal names.
        if use_mapping and self.__region_map:
            pat_id = self.__series.study.patient.id
            study_id = self.__series.study.id
            def map_name(info):
                info['name'], _ = self.__region_map.to_internal(info['name'], pat_id=pat_id, study_id=study_id)
                return info
            roi_info = dict((id, map_name(info)) for id, info in roi_info.items())

        return roi_info

    def has_landmark(
        self,
        landmark: Landmark,
        token: str = 'Marker') -> bool:
        return landmark in self.list_landmarks(token=token)

    def has_regions(
        self,
        region: Region,
        use_mapping: bool = True) -> bool:
        return region in self.list_regions(regions=region, use_mapping=use_mapping)

    def landmark_data(
        self,
        data_only: bool = False,
        landmarks: Landmarks = 'all',
        token: str = 'Marker',
        use_patient_coords: bool = True,
        **kwargs) -> Optional[pd.DataFrame]:
        landmarks = regions_to_list(landmarks, literals={ 'all': lambda: self.list_landmarks(token=token) })
        rtstruct_dicom = self.dicom
        lms = []
        for l in landmarks:
            lm = RtStructConverter.get_roi_landmark(rtstruct_dicom, l)
            if not use_patient_coords:
                spacing = self.ref_ct.spacing
                offset = self.ref_ct.offset
                lm = (lm - offset) / spacing
                lm = lm.round()
                lm = lm.astype(np.uint32)
            lms.append(lm)
        if len(lms) == 0:
            return None
        lms = np.vstack(lms)
        lm_df = pd.DataFrame(lms, index=landmarks).reset_index()
        lm_df = lm_df.rename(columns={ 'index': 'landmark-id' })

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in lm_df.columns:
            lm_df.insert(0, 'patient-id', self.__series.study.patient.id)
        if 'study-id' not in lm_df.columns:
            lm_df.insert(1, 'study-id', self.__series.study.id)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        lm_df = lm_df.sort_values(['patient-id', 'study-id', 'landmark-id'])

        if data_only:
            return lm_df[range(3)].to_numpy().astype(np.float32)
        else:
            return lm_df

    def list_landmarks(
        self,
        token: str = 'Marker') -> List[str]:
        lms = self.list_regions(landmarks_token=None)
        lms = [l for l in lms if token in l]
        return lms

    def list_regions(
        self,
        # Only the regions in 'regions' should be returned, saves us from performing filtering code elsewhere.
        landmarks_token: Optional[str] = 'Marker',
        regions: Optional[Regions] = 'all',
        return_unmapped: bool = False,
        use_mapping: bool = True) -> Union[List[Region], Tuple[List[Region], List[Region]]]:
        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Get unmapped region names.
        rtstruct_dicom = self.dicom
        unmapped_regions = RtStructConverter.get_roi_names(rtstruct_dicom)

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        unmapped_regions = list(filter(lambda r: RtStructConverter.has_roi_data(rtstruct_dicom, r), unmapped_regions))

        # Map regions using 'region-map.csv'.
        if use_mapping:
            pat_id = self.__series.study.patient.id
            study_id = self.__series.study.id
            # Store as ('unmapped region', 'mapped region', 'priority').
            mapped_regions = []
            for unmapped_region in unmapped_regions:
                mapped_region, priority = self.__region_map.to_internal(unmapped_region, pat_id=pat_id, study_id=study_id)
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

        # Filter on 'regions'. If region mapping is used (i.e. mapped_regions != None),
        # this will try to match mapped names, otherwise it will map unmapped names.
        if regions != 'all':
            regions = regions_to_list(regions)

            if use_mapping:
                mapped_regions = [r for r in mapped_regions if r[1] in regions]
            else:
                unmapped_regions = [r for r in unmapped_regions if r in regions]

        # Check for multiple regions.
        if not self.__series.region_policy['duplicates']['allow']:
            if use_mapping:
                # Only check for duplicates on mapped regions.
                names = [r[1] for r in mapped_regions]
            else:
                names = unmapped_regions

            # Get duplicated regions.
            dup_regions = [r for r in names if names.count(r) > 1]

            if len(dup_regions) > 0:
                if use_mapping and self.__region_map is not None:
                    raise ValueError(f"Duplicate regions found for RtStruct '{self}', perhaps a 'region-map.csv' issue? Duplicated regions: '{dup_regions}'")
                else:
                    raise ValueError(f"Duplicate regions found for RtStruct '{self}'. Duplicated regions: '{dup_regions}'")

        # Sort regions.
        if use_mapping:
            mapped_regions = list(sorted(mapped_regions, key=lambda r: r[1]))
        else:
            unmapped_regions = list(sorted(unmapped_regions))

        # Filter landmarks.
        if landmarks_token is not None:
            if use_mapping:
                mapped_regions = list(filter(lambda r: landmarks_token not in r[1], mapped_regions))
            else:
                unmapped_regions = list(filter(lambda r: landmarks_token not in r, unmapped_regions))

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
        regions: Regions = 'all',    # Request specific region/s, otherwise get all region data. Specific regions must exist.
        regions_ignore_missing: bool = False,
        use_mapping: bool = True,
        **kwargs) -> OrderedDict:

        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Check that RTSTRUCT has requested 'regions'.
        regions = regions_to_list(regions, literals={ 'all': self.list_regions })
        rtstruct_regions = self.list_regions(regions=regions, use_mapping=use_mapping)
        if not regions_ignore_missing:
            for r in regions:
                if not r in rtstruct_regions:
                    raise ValueError(f"Requested region '{r}' not present for RtStruct '{self}'.")

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        rtstruct_regions = self.list_regions(return_unmapped=True, use_mapping=use_mapping)

        # Filter on requested regions.
        if use_mapping:
            rtstruct_regions = [r for r in rtstruct_regions if r[1] in regions]
        else:
            rtstruct_regions = [r for r in rtstruct_regions if r in regions]

        # Load RTSTRUCT dicom.
        rtstruct_dicom = self.dicom

        # Get region data.
        data = {}

        if use_mapping:
            # Load region using unmapped name, store using mapped name.
            for unmapped_region, mapped_region in rtstruct_regions:
                rdata = RtStructConverter.get_region_data(rtstruct_dicom, unmapped_region, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.offset)
                data[mapped_region] = rdata
        else:
            # Load and store region using unmapped name.
            for r in rtstruct_regions:
                rdata = RtStructConverter.get_region_data(rtstruct_dicom, r, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.offset)
                data[r] = rdata

        # Sort dict keys.
        data = collections.OrderedDict((n, data[n]) for n in sorted(data.keys())) 

        return data

    def __load_data(self) -> None:
        if not self.__series.index_policy['no-ref-ct']['allow']:
            # Get referenced CT series from index.
            ct_series_id = self.__index['mod-spec'][DICOM_RTSTRUCT_REF_CT_KEY]
            self.__ref_ct = CtSeries(self.__series.study, ct_series_id)
        else:
            # Choose last CT series in study as "ref".
            self.__ref_ct = self.__series.study.default_ct

    def __str__(self) -> str:
        return self.__global_id
