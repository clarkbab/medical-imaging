import collections
import numpy as np
import pandas as pd
import pydicom as dcm
from typing import *

from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *

# Must import from series submodules to avoid circular import.
from ...series.ct import CtSeries
from ...series.series import Modality
from ..files import DicomFile, SOPInstanceUID
from .region_map import RegionMap
from .rtstruct_converter import RtStructConverter

class RtStructFile(DicomFile):
    def __init__(
        self,
        series: 'RtStructSeries',
        id: SOPInstanceUID,
        region_dups: Optional[pd.DataFrame] = None,
        region_map: Optional[RegionMap] = None):
        self.__global_id = f"{series}:{id}"
        self.__id = id
        self.__ref_ct = None        # Lazy-loaded.
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

    @property
    def rtstruct(self) -> dcm.dataset.FileDataset:
        return dcm.read_file(self.__path)
    
    @property
    def series(self) -> 'RtStructSeries':
        return self.__series

    def get_region_info(
        self,
        use_mapping: bool = True) -> Dict[int, Dict[str, str]]:
        # Load RTSTRUCT dicom.
        rtstruct = self.rtstruct

        # Get region IDs.
        roi_info = RtStructConverter.get_roi_info(rtstruct)

        # Filter names on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        roi_info = dict(filter(lambda i: RtStructConverter.has_roi_data(rtstruct, i[1]['name']), roi_info.items()))

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
        landmarks: Landmarks = 'all',
        token: str = 'Marker',
        use_patient_coords: bool = True,
        **kwargs) -> Optional[pd.DataFrame]:
        landmarks = regions_to_list(landmarks, literals={ 'all': lambda: self.list_landmarks(token=token) })
        rtstruct = self.rtstruct
        lms = []
        for l in landmarks:
            lm = RtStructConverter.get_roi_landmark(rtstruct, l)
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

        return lm_df

    def list_landmarks(
        self,
        token: str = 'Marker') -> List[str]:
        lms = self.list_regions(landmarks_token=None)
        lms = [int(l) for l in lms if token in l]
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
        rtstruct = self.rtstruct
        unmapped_regions = RtStructConverter.get_roi_names(rtstruct)

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        unmapped_regions = list(filter(lambda r: RtStructConverter.has_roi_data(rtstruct, r), unmapped_regions))

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
                    raise ValueError(f"Duplicate regions found for RtStructFile '{self}', perhaps a 'region-map.csv' issue? Duplicated regions: '{dup_regions}'")
                else:
                    raise ValueError(f"Duplicate regions found for RtStructFile '{self}'. Duplicated regions: '{dup_regions}'")

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
                    raise ValueError(f"Requested region '{r}' not present for RtStructFile '{self}'.")

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        rtstruct_regions = self.list_regions(return_unmapped=True, use_mapping=use_mapping)

        # Filter on requested regions.
        if use_mapping:
            rtstruct_regions = [r for r in rtstruct_regions if r[1] in regions]
        else:
            rtstruct_regions = [r for r in rtstruct_regions if r in regions]

        # Get reference CTs.
        cts = self.ref_ct.ct_files

        # Load RTSTRUCT dicom.
        rtstruct = self.rtstruct

        # Get region data.
        data = {}
        if use_mapping:
            # Load region using unmapped name, store using mapped name.
            for unmapped_region, mapped_region in rtstruct_regions:
                rdata = RtStructConverter.get_roi_contour(rtstruct, unmapped_region, cts)
                data[mapped_region] = rdata
        else:
            # Load and store region using unmapped name.
            for r in rtstruct_regions:
                rdata = RtStructConverter.get_roi_contour(rtstruct, r, cts)
                data[r] = rdata

        # Sort dict keys.
        data = collections.OrderedDict((n, data[n]) for n in sorted(data.keys())) 

        return data

    def __verify_index(self) -> None:
        if len(self.__index) == 0:
            raise ValueError(f"RtStructFile '{self}' not found in index for series '{self.__series}'.")
        elif len(self.__index) > 1:
            raise ValueError(f"Multiple RtStructFiles found in index with SOPInstanceUID '{self.__id}' for series '{self.__series}'.")

    def __load_ref_ct(self) -> None:
        if not self.__index_policy['no-ref-ct']['allow']:
            # Get CT series referenced in RTSTRUCT DICOM.
            rtstruct = self.rtstruct
            ct_id = rtstruct.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID

        elif self.__index_policy['no-ref-ct']['only'] == 'at-least-one-ct' or self.__index_policy['no-ref-ct']['only'] == 'single-ct':
            # Load first CT series in study.
            ct_id = self.__series.study.list_series('CT')[-1]

        self.__ref_ct = CtSeries(self.__series.study, ct_id)

    def __str__(self) -> str:
        return self.__global_id
