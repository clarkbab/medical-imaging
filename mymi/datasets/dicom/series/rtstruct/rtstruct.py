import collections
import os
import pandas as pd
from typing import *

from mymi import config
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from ....region_map import RegionMap
from ..ct import CtSeries
from ..series import DicomSeries
from .rtstruct_converter import RtStructConverter

LANDMARKS_REGEXP = r'^Marker \d+$'

class RtStructSeries(DicomSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: SeriesID,
        ref_ct: CtSeries,
        index: pd.Series,
        index_policy: Dict[str, Any],
        region_map: Optional[RegionMap] = None) -> None:
        datasetpath = os.path.join(config.directories.datasets, 'dicom', dataset_id, 'data', 'patients')
        self.__dataset_id = dataset_id
        self.__filepath = os.path.join(datasetpath, index['filepath'])
        self.__id = id
        self.__index = index
        self.__index_policy = index_policy
        self.__modality = 'rtstruct'
        self.__pat_id = pat_id
        self.__ref_ct = ref_ct
        self.__region_map = region_map
        self.__study_id = study_id

    @property
    def dicom(self) -> CtDicom:
        return dcm.dcmread(self.__filepath)

    def has_landmark(
        self,
        landmark_ids: LandmarkIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmark_ids = arg_to_list(landmark_ids, LandmarkID)
        n_overlap = len(np.intersect1d(landmark_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmark_ids)

    def has_region(
        self,
        region_ids: RegionIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_regions(**kwargs)
        region_ids = arg_to_list(region_ids, RegionID)
        n_overlap = len(np.intersect1d(region_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(region_ids)

    def landmark_data(
        self,
        data_only: bool = False,
        landmark_ids: LandmarkIDs = 'all',
        landmarks_regexp: str = LANDMARKS_REGEXP,
        show_ids: bool = True,
        use_patient_coords: bool = True,
        **kwargs) -> Optional[Union[LandmarksData, LandmarksDataVox, Points3D, Voxels]]:
        # Load landmarks.
        landmarks = self.list_landmarks(landmark_ids=landmark_ids, landmarks_regexp=landmarks_regexp, **kwargs)
        rtstruct_dicom = self.dicom
        lms = []
        for l in landmarks:
            lm = RtStructConverter.get_roi_landmark(rtstruct_dicom, l)
            lms.append(lm)
        if len(lms) == 0:
            return None
        
        # Convert to DataFrame.
        lms = np.vstack(lms)
        landmark_data = pd.DataFrame(lms, index=landmarks).reset_index()
        landmark_data = landmark_data.rename(columns={ 'index': 'landmark-id' })
        if not use_patient_coords:
            landmark_data = landmarks_to_image_coords(landmark_data, self.ref_ct.spacing, self.ref_ct.offset)

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if show_ids:
            if 'patient-id' not in landmark_data.columns:
                landmark_data.insert(0, 'patient-id', self.__pat_id)
            if 'study-id' not in landmark_data.columns:
                landmark_data.insert(1, 'study-id', self.__study_id)
            if 'series-id' not in landmark_data.columns:
                landmark_data.insert(2, 'series-id', self.__id)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        sort_cols = []
        if 'patient-id' in landmark_data.columns:
            sort_cols += ['patient-id']
        if 'study-id' in landmark_data.columns:
            sort_cols += ['study-id']
        if 'series-id' in landmark_data.columns:
            sort_cols += ['series-id']
        sort_cols += ['landmark-id']
        landmark_data = landmark_data.sort_values(sort_cols)

        if data_only:
            return landmarks_to_data(landmark_data)
        else:
            return landmark_data

    def list_landmarks(
        self,
        landmark_ids: Landmarks = 'all', 
        landmarks_regexp: str = LANDMARKS_REGEXP) -> List[LandmarkID]:
        ids = self.list_regions(landmarks_regexp=None)
        ids = [l for l in ids if re.match(landmarks_regexp, l) is not None in l]
        if landmark_ids != 'all':
            ids = [i for i in ids if i in regions_to_list(landmark_ids)]
        return ids

    def list_regions(
        self,
        landmarks_regexp: Optional[str] = LANDMARKS_REGEXP,
        region_ids: RegionIDs = 'all',
        return_numbers: bool = False,
        return_unmapped: bool = False,
        use_mapping: bool = True) -> Union[List[RegionID], Tuple[List[RegionID], List[RegionID]], Tuple[List[RegionID], List[int]], Tuple[List[RegionID], List[RegionID], List[int]]]:
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
            unmapped_ids = []   # We need to return these so 'region_data' can load from rtstruct.
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

        # Filter on 'region_ids'. If region mapping is used (i.e. mapped_regions != None),
        # this will try to match mapped names, otherwise it will map unmapped names.
        if region_ids != 'all':
            region_ids = regions_to_list(region_ids)
            if use_mapping:
                if return_numbers:
                    mapped_ids, unmapped_ids, numbers = filter_lists([mapped_ids, unmapped_ids, numbers], lambda i: i[0] in region_ids)
                else:
                    mapped_ids, unmapped_ids = filter_lists([mapped_ids, unmapped_ids], lambda i: i[0] in region_ids)
            else:
                unmapped_ids = [r for r in unmapped_ids if r in region_ids]

        # Filter out landmarks based on 'landmarks_regexp'.
        if landmarks_regexp is not None:
            if use_mapping:
                if return_numbers:
                    mapped_ids, unmapped_ids, numbers = filter_lists([mapped_ids, unmapped_ids, numbers], lambda i: landmarks_regexp not in i[0])
                else:
                    mapped_ids, unmapped_ids = filter_lists([mapped_ids, unmapped_ids], lambda i: landmarks_regexp not in i[0])
            else:
                unmapped_ids = [r for r in unmapped_ids if landmarks_regexp not in r]

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

    def region_data(
        self,
        region_ids: RegionID = 'all',
        regions_ignore_missing: bool = False,
        use_mapping: bool = True,
        **kwargs) -> RegionsData:

        # If not 'region-map.csv' exists, set 'use_mapping=False'.
        if self.__region_map is None:
            use_mapping = False

        # Get patient regions. If 'use_mapping=True', return unmapped region names too - we'll
        # need these to load regions from RTSTRUCT dicom.
        if use_mapping:
            mapped_ids, unmapped_ids = self.list_regions(region_ids=region_ids, return_unmapped=True)
        else:
            unmapped_ids = self.list_regions(region_ids=region_ids, use_mapping=False)

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
                    rdata = RtStructConverter.get_region_data(rtstruct_dicom, c, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.offset)
                    labels.append(rdata)
                label = np.maximum(*labels) if len(labels) > 1 else labels[0]
                data[m] = label
        else:
            # Load and store region using unmapped name.
            for u in unmapped_ids:
                rdata = RtStructConverter.get_region_data(rtstruct_dicom, u, self.ref_ct.size, self.ref_ct.spacing, self.ref_ct.offset)
                data[u] = rdata

        # Sort dict keys.
        data = collections.OrderedDict((n, data[n]) for n in sorted(data.keys())) 

        return data

# Add properties.
props = ['dataset_id', 'filepath', 'id', 'index', 'index_policy', 'modality', 'pat_id', 'ref_ct', 'region_map', 'study_id']
for p in props:
    setattr(RtStructSeries, p, property(lambda self, p=p: getattr(self, f'_{RtStructSeries.__name__}__{p}')))
