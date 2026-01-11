import collections
import os
import pandas as pd
from typing import *

from mymi import config as conf
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ....region_map import RegionMap
from ..ct import DicomCtSeries
from ..series import DicomSeries
from .rtstruct_converter import RtStructConverter

DEFAULT_LANDMARK_REGEXP = r'^Marker \d+$'

class DicomRtStructSeries(DicomSeries):
    def __init__(
        self,
        dataset: 'DicomDataset',
        pat: 'DicomPatient',
        study: 'DicomStudy',
        id: SeriesID,
        ref_ct: DicomCtSeries,
        index: pd.Series,
        index_policy: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        region_map: Optional[RegionMap] = None) -> None:
        super().__init__('rtstruct', dataset, pat, study, id, config=config)
        self.__filepath = os.path.join(conf.directories.datasets, 'dicom', dataset.id, 'data', 'patients', index['filepath'])
        self.__modality = 'rtstruct'
        self.__ref_ct = ref_ct
        self.__region_map = region_map

    @property
    def dicom(self) -> CtDicom:
        return dcm.dcmread(self.__filepath)

    @property
    def filepath(self) -> FilePath:
        return self.__filepath

    def has_landmark(
        self,
        landmark: LandmarkIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmarks = arg_to_list(landmark, LandmarkID, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(landmarks, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmarks)

    def has_region(
        self,
        region: RegionIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_regions(**kwargs)
        regions = arg_to_list(region, RegionID)
        n_overlap = len(np.intersect1d(regions, all_ids))
        return n_overlap > 0 if any else n_overlap == len(regions)

    @property
    def landmark_regexp(self) -> str:
        if self._config is not None and 'landmarks' in self._config and 'regexp' in self._config['landmarks']:
            return self._config['landmarks']['regexp']
        else:
            return DEFAULT_LANDMARK_REGEXP

    def landmarks_data(
        self,
        points_only: bool = False,
        landmark: LandmarkIDs = 'all',
        landmark_regexp: Optional[str] = None,
        n: Optional[int] = None,
        show_ids: bool = True,
        use_patient_coords: bool = True,
        **kwargs) -> Optional[Union[LandmarksFrame, LandmarksFrameVox, Points3D, Voxels]]:
        # Load landmarks.
        landmarks = self.list_landmarks(landmark=landmark, landmark_regexp=landmark_regexp, **kwargs)
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
            landmarks_data = landmarks_to_image_coords(landmarks_data, self.ref_ct.spacing, self.ref_ct.origin)

        # Filter by number of rows.
        if n is not None:
            landmarks_data = landmarks_data.iloc[:n]

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if show_ids:
            if 'patient-id' not in landmarks_data.columns:
                landmarks_data.insert(0, 'patient-id', self._pat_id)
            if 'study-id' not in landmarks_data.columns:
                landmarks_data.insert(1, 'study-id', self._study_id)
            if 'series-id' not in landmarks_data.columns:
                landmarks_data.insert(2, 'series-id', self._id)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        sort_cols = []
        if 'patient-id' in landmarks_data.columns:
            sort_cols += ['patient-id']
        if 'study-id' in landmarks_data.columns:
            sort_cols += ['study-id']
        if 'series-id' in landmarks_data.columns:
            sort_cols += ['series-id']
        sort_cols += ['landmark-id']
        landmarks_data = landmarks_data.sort_values(sort_cols)

        if points_only:
            return landmarks_to_data(landmarks_data)
        else:
            return landmarks_data

    def list_landmarks(
        self,
        landmark: LandmarkIDs = 'all', 
        landmark_regexp: Optional[str] = None) -> List[LandmarkID]:
        if landmark_regexp is None:
            landmark_regexp = self.landmark_regexp
        ids = self.list_regions(filter_landmarks=False)
        # Both landmarks/regions are stored in rtstruct, but we only want objects like 'Marker 1' for example.
        ids = [l for l in ids if re.match(landmark_regexp, l)]
        if landmark != 'all':
            ids = [i for i in ids if i in regions_to_list(landmark)]
        return ids

    # 1. Should return only regions when landmark_regexp is None, load regexp from config or default.
    # 2. Should return landmarks also when use_landmark_regexp is False.

    def list_regions(
        self,
        filter_landmarks: bool = True,
        landmark_regexp: Optional[str] = None,
        region: RegionIDs = 'all',
        return_numbers: bool = False,
        return_unmapped: bool = False,
        use_mapping: bool = True) -> Union[List[RegionID], Tuple[List[RegionID], List[RegionID]], Tuple[List[RegionID], List[int]], Tuple[List[RegionID], List[RegionID], List[int]]]:
        # Get the landmark regexp - used to filter out landmarks.
        if filter_landmarks and landmark_regexp is None:
            landmark_regexp = self.landmark_regexp

        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Get unmapped region names.
        rtstruct_dicom = self.dicom
        ids = RtStructConverter.get_roi_names(rtstruct_dicom)
        if return_numbers:
            nums = RtStructConverter.get_roi_numbers(rtstruct_dicom)

        # Filter regions on those for which data can be obtained, e.g. some may not have
        # 'ContourData' and shouldn't be included.
        if return_numbers:
            ids, nums = filter_lists([ids, nums], lambda i: RtStructConverter.has_roi_data(rtstruct_dicom, i[0]))
        else:
            ids = list(filter(lambda i: RtStructConverter.has_roi_data(rtstruct_dicom, i), ids))

        # Map regions using 'region-map.csv'.
        if use_mapping:
            mapped_ids = []
            unmapped_ids = []   # We need to return these so 'regions_data' can load from rtstruct.
            if return_numbers:
                numbers = []    # We might need to return roi numbers for dose calculation or other applications that require rtstruct access.
            for i, id in enumerate(ids):
                mapped_id = self.__region_map.map_region(id)

                if mapped_id == id:  # No mapping occurred.
                    mapped_ids.append(id)
                    unmapped_ids.append(id)
                    if return_numbers:
                        numbers.append(nums[i])
                elif mapped_id in ids: # Mapped region would clash with an existing region in the rtstruct.
                    logging.warning(f"Mapped region '{mapped_id}' (mapped from '{id}') already found in unmapped regions for '{self}'. Skipping.")
                    continue
                # # Don't map regions that are already present in 'new_regions'.
                # elif mapped_id in mapped_ids:
                #     raise ValueError(f"Mapped region '{mapped_id}' (mapped from '{i}') already found in mapped regions for '{self}'. Set 'priority' in region map.")
                # Allow multiple regions to be mapped to the same region (e.g. Chestwall_L/R -> Chestwall)
                # and combine these regions into one super-region.
                elif mapped_id in mapped_ids:   # A value has already been mapped to this region.
                    # Add this to the existing list of unmapped ids for this mapped_id.
                    idx = mapped_ids.index(mapped_id)
                    new_ids = arg_to_list(unmapped_ids[idx], RegionID) + [id]
                    unmapped_ids[idx] = new_ids
                    if return_numbers:
                        new_nums = arg_to_list(numbers[idx], int) + [nums[i]]
                        numbers[idx] = new_nums
                else:
                    # Mapping without issues and tissues.
                    mapped_ids.append(mapped_id)
                    unmapped_ids.append(id)
                    if return_numbers:
                        numbers.append(nums[i])
        else:
            unmapped_ids = ids
            if return_numbers:
                numbers = nums

        # Filter on 'region'. If region mapping is used (i.e. mapped_regions != None),
        # this will try to match mapped names, otherwise it will map unmapped names.
        if region != 'all':
            regions = regions_to_list(region)
            if use_mapping:
                if return_numbers:
                    mapped_ids, unmapped_ids, numbers = filter_lists([mapped_ids, unmapped_ids, numbers], lambda i: i[0] in regions)
                else:
                    mapped_ids, unmapped_ids = filter_lists([mapped_ids, unmapped_ids], lambda i: i[0] in regions)
            else:
                unmapped_ids = [r for r in unmapped_ids if r in regions]

        # Filter out landmarks based on 'landmark_regexp'.
        if filter_landmarks and landmark_regexp is not None:
            if use_mapping:
                if return_numbers:
                    mapped_ids, unmapped_ids, numbers = filter_lists([mapped_ids, unmapped_ids, numbers], lambda i: not re.match(landmark_regexp, i[0]))
                else:
                    mapped_ids, unmapped_ids = filter_lists([mapped_ids, unmapped_ids], lambda i: not re.match(landmark_regexp, i[0]))
            else:
                unmapped_ids = [r for r in unmapped_ids if not re.match(landmark_regexp, r)]

        # Sort regions.
        if use_mapping:
            if return_numbers:
                mapped_ids, unmapped_ids, numbers = sort_lists([mapped_ids, unmapped_ids, numbers], lambda i: i[0])
            else:
                mapped_ids, unmapped_ids = sort_lists([mapped_ids, unmapped_ids], lambda i: i[0])
        else:
            unmapped_ids = list(sorted(unmapped_ids))

        # Choose return type when using mapping.
        if use_mapping:
            if return_unmapped:
                return (mapped_ids, unmapped_ids, numbers) if return_numbers else (mapped_ids, unmapped_ids)
            else:
                return (mapped_ids, numbers) if return_numbers else mapped_ids
        elif return_numbers:
            return unmapped_ids, numbers
        else:
            return unmapped_ids
        
    @property
    def ref_ct(self) -> DicomCtSeries:
        return self.__ref_ct

    def regions_data(
        self,
        region: RegionID = 'all',
        regions_ignore_missing: bool = False,
        use_mapping: bool = True,
        **kwargs) -> RegionArrays:

        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        if use_mapping:
            mapped_ids, unmapped_ids = self.list_regions(region=region, return_unmapped=True)
        else:
            unmapped_ids = self.list_regions(region=region, use_mapping=False)

        # Load data from dicom.
        rtstruct_dicom = self.dicom
        data = {}
        if use_mapping:
            # Load region using unmapped name, store using mapped name.
            for m, u in zip(mapped_ids, unmapped_ids):
                # Multiple regions could be mapped to the same region, e.g. Chestwall_L/R -> Chestwall.
                combine_ids = arg_to_list(u, RegionID)
                labels = []
                for c in combine_ids:
                    rdata = RtStructConverter.get_regions_data(rtstruct_dicom, c, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.origin)
                    labels.append(rdata)
                label = np.maximum(*labels) if len(labels) > 1 else labels[0]
                data[m] = label
        else:
            # Load and store region using unmapped name.
            for u in unmapped_ids:
                rdata = RtStructConverter.get_regions_data(rtstruct_dicom, u, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.origin)
                data[u] = rdata

        # Sort dict keys.
        data = collections.OrderedDict((n, data[n]) for n in sorted(data.keys())) 

        return data

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
