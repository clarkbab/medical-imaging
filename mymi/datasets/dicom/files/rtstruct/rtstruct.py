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
    def dicom(self) -> CtDicom:
        return dcm.read_file(self.__filepath)

    @property
    @ensure_loaded
    def ref_ct(self) -> CtSeries:
        return self.__ref_ct

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

    def has_landmarks(
        self,
        landmark_ids: LandmarkIDs = 'all',
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_landmarks(landmark_ids=landmark_ids, **kwargs)
        req_ids = regions_to_list(landmark_ids)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def has_regions(
        self,
        region_ids: RegionIDs,
        any: bool = False,
        **kwargs) -> bool:
        real_ids = self.list_regions(region_ids=region_ids, **kwargs)
        req_ids = regions_to_list(region_ids)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def landmarks_data(
        self,
        data_only: bool = False,
        landmark_ids: LandmarkIDs = 'all',
        token: str = 'Marker',
        use_patient_coords: bool = True,
        **kwargs) -> Optional[Union[LandmarksData, LandmarksVoxelData, Points3D, Voxels]]:
        # Load landmarks.
        landmarks = self.list_landmarks(landmark_ids=landmark_ids, token=token)
        rtstruct_dicom = self.dicom
        lms = []
        for l in landmarks:
            lm = RtStructConverter.get_roi_landmark(rtstruct_dicom, l)
            lms.append(lm)
        if len(lms) == 0:
            return None
        
        # Convert to DataFrame.
        lms = np.vstack(lms)
        landmarks_data = pd.DataFrame(lms, index=landmarks).reset_index()
        landmarks_data = landmarks_data.rename(columns={ 'index': 'landmark-id' })
        if not use_patient_coords:
            landmarks_data = landmarks_to_image_coords(landmarks_data, self.ref_ct.spacing, self.ref_ct.offset)

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in landmarks_data.columns:
            landmarks_data.insert(0, 'patient-id', self.__series.study.patient.id)
        if 'study-id' not in landmarks_data.columns:
            landmarks_data.insert(1, 'study-id', self.__series.study.id)
        if 'series-id' not in landmarks_data.columns:
            landmarks_data.insert(2, 'series-id', self.__series.id)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        landmarks_data = landmarks_data.sort_values(['patient-id', 'study-id', 'series-id', 'landmark-id'])

        if data_only:
            return landmarks_data[range(3)].to_numpy().astype(float)
        else:
            return landmarks_data

    def list_landmarks(
        self,
        landmark_ids: Landmarks = 'all', 
        token: str = 'Marker') -> List[LandmarkID]:
        ids = self.list_regions(landmarks_token=None)
        ids = [l for l in ids if token in l]
        if landmark_ids != 'all':
            ids = [i for i in ids if i in regions_to_list(landmark_ids)]
        return ids

    def list_regions(
        self,
        landmarks_token: Optional[str] = 'Marker',
        region_ids: RegionIDs = 'all',
        return_unmapped: bool = False,
        use_mapping: bool = True) -> Union[List[RegionID], Tuple[List[RegionID], List[RegionID]]]:
        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Get unmapped region names.
        rtstruct_dicom = self.dicom
        unmapped_ids = RtStructConverter.get_roi_names(rtstruct_dicom)

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        unmapped_ids = list(filter(lambda r: RtStructConverter.has_roi_data(rtstruct_dicom, r), unmapped_ids))

        # Map regions using 'region-map.csv'.
        if use_mapping:
            pat_id = self.__series.study.patient.id
            study_id = self.__series.study.id
            # Store as ('unmapped region', 'mapped region', 'priority').
            mapped_ids = []
            for unmapped_id in unmapped_ids:
                mapped_id, priority = self.__region_map.to_internal(unmapped_id, pat_id=pat_id, study_id=study_id)
                # Don't map regions that would map to an existing region name.
                if mapped_id != unmapped_id and mapped_id in unmapped_ids:
                    logging.warning(f"Mapped RegionID '{mapped_id}' (mapped from '{unmapped_id}') already found in unmapped regions for '{self}'. Skipping.")
                    mapped_ids.append((unmapped_id, mapped_id, priority))

                # Don't map regions that are already present in 'new_regions'.
                elif mapped_id in mapped_ids:
                    raise ValueError(f"Mapped RegionID '{mapped_id}' (mapped from '{unmapped_id}') already found in mapped regions for '{self}'. Set 'priority' in region map.")

                # Map region.
                else:
                    mapped_ids.append((unmapped_id, mapped_id, priority))

            # If multiple unmapped regions map to the same region, then choose to map the one with
            # higher priorty.
            for i in range(len(mapped_ids)):
                unmapped_id, mapped_id, priority = mapped_ids[i]
                for _, mr, p in mapped_ids:
                    # If another mapped region exists with a higher priority, then set this region
                    # back to its unmapped form.
                    if mr == mapped_id and p > priority:
                        mapped_ids[i] = (unmapped_id, unmapped_id, priority)

            # Remove priority.
            mapped_ids = [r[:-1] for r in mapped_ids]

        # Filter on 'regions'. If region mapping is used (i.e. mapped_ids != None),
        # this will try to match mapped names, otherwise it will map unmapped names.
        if region_ids != 'all':
            region_ids = regions_to_list(region_ids)
            if use_mapping:
                mapped_ids = [r for r in mapped_ids if r[1] in region_ids]
            else:
                unmapped_ids = [r for r in unmapped_ids if r in region_ids]

        # Check for multiple regions.
        if not self.__series.region_policy['duplicates']['allow']:
            if use_mapping:
                # Only check for duplicates on mapped regions.
                names = [r[1] for r in mapped_ids]
            else:
                names = unmapped_ids

            # Get duplicated regions.
            dup_regions = [r for r in names if names.count(r) > 1]

            if len(dup_regions) > 0:
                if use_mapping and self.__region_map is not None:
                    raise ValueError(f"Duplicate regions found for RtStruct '{self}', perhaps a 'region-map.csv' issue? Duplicated regions: '{dup_regions}'")
                else:
                    raise ValueError(f"Duplicate regions found for RtStruct '{self}'. Duplicated regions: '{dup_regions}'")

        # Sort regions.
        if use_mapping:
            mapped_ids = list(sorted(mapped_ids, key=lambda r: r[1]))
        else:
            unmapped_ids = list(sorted(unmapped_ids))

        # Filter landmarks.
        if landmarks_token is not None:
            if use_mapping:
                mapped_ids = list(filter(lambda r: landmarks_token not in r[1], mapped_ids))
            else:
                unmapped_ids = list(filter(lambda r: landmarks_token not in r, unmapped_ids))

        # Choose return type when using mapping.
        if use_mapping:
            if return_unmapped:
                return mapped_ids
            else:
                mapped_ids = [r[1] for r in mapped_ids]
                return mapped_ids
        else:
            return unmapped_ids

    def regions_data(
        self,
        region_ids: RegionIDs = 'all',    # Request specific region/s, otherwise get all region data. Specific regions must exist.
        regions_ignore_missing: bool = True,
        use_mapping: bool = True,
        **kwargs) -> RegionsData:

        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Check that RTSTRUCT has requested 'region_ids'.
        region_ids = regions_to_list(region_ids, literals={ 'all': self.list_regions })
        rtstruct_region_ids = self.list_regions(region_ids=region_ids, use_mapping=use_mapping)
        if not regions_ignore_missing:
            for r in region_ids:
                if not r in rtstruct_region_ids:
                    raise ValueError(f"Requested RegionID '{r}' not present for RtStruct '{self}'.")

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        rtstruct_region_ids = self.list_regions(return_unmapped=True, use_mapping=use_mapping)

        # Filter on requested regions.
        if use_mapping:
            rtstruct_region_ids = [r for r in rtstruct_region_ids if r[1] in region_ids]
        else:
            rtstruct_region_ids = [r for r in rtstruct_region_ids if r in region_ids]

        # Load RTSTRUCT dicom.
        rtstruct_dicom = self.dicom

        # Get region data.
        data = {}

        if use_mapping:
            # Load region using unmapped name, store using mapped name.
            for unmapped_id, mapped_id in rtstruct_region_ids:
                rdata = RtStructConverter.get_regions_data(rtstruct_dicom, unmapped_id, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.offset)
                data[mapped_id] = rdata
        else:
            # Load and store region using unmapped name.
            for r in rtstruct_region_ids:
                rdata = RtStructConverter.get_regions_data(rtstruct_dicom, r, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.offset)
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

# Add properties.
props = ['filepath', 'global_id', 'id', 'index', 'series']
for p in props:
    setattr(RtStructFile, p, property(lambda self, p=p: getattr(self, f'_{RtStructFile.__name__}__{p}')))
