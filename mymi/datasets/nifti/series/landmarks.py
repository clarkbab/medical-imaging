import numpy as np
import os
from typing import *

from mymi.transforms import sample
from mymi.typing import *
from mymi.utils import *

from .series import NiftiSeries

class LandmarksSeries(NiftiSeries):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__global_id = f'{study}:{id}'
        self.__filepath = os.path.join(study.path, 'landmarks', f'{id}.csv')
        self.__study = study

    @property
    def id(self) -> SeriesID:
        return self.__id

    @property
    def path(self) -> str:
        return self.__path

    @property
    def study(self) -> str:
        return self.__study

    def data(
        self,
        data_only: bool = False,
        landmark_ids: LandmarkIDs = 'all',
        sample_ct: bool = False,
        sample_dose: bool = False,
        use_patient_coords: bool = True,
        **kwargs) -> Union[LandmarksData, LandmarksVoxelData, Points3D, Voxels]:

        # Load landmarks.
        landmarks_data = load_csv(self.__filepath)
        landmarks_data = landmarks_data.rename(columns={ '0': 0, '1': 1, '2': 2 })
        if not use_patient_coords:
            landmarks_data = landmarks_to_image_coords(landmarks_data, self.__study.ct_spacing, self.__study.ct_offset)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        landmarks_data = landmarks_data.sort_values('landmark-id')

        if landmark_ids != 'all':
            landmark_ids = self.list_landmarks(landmark_ids=landmark_ids)
            landmarks_data = landmarks_data[landmarks_data['landmark-id'].isin(landmark_ids)]

        # Add sampled CT intensities.
        if sample_ct:
            ct_image = self.__study.default_ct
            ct_values = sample(ct_image.data, landmarks_to_data(landmarks_data), spacing=ct_image.spacing, offset=ct_image.offset, **kwargs)
            landmarks_data['ct'] = ct_values

        # Add sampled dose intensities.
        if sample_dose:
            dose_image = self.__study.default_dose
            dose_values = sample(dose_image.data, landmarks_to_data(landmarks_data), spacing=dose_image.spacing, offset=dose_image.offset, **kwargs)
            landmarks_data['dose'] = dose_values

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in landmarks_data.columns:
            landmarks_data.insert(0, 'patient-id', self.study.patient.id)
        if 'study-id' not in landmarks_data.columns:
            landmarks_data.insert(1, 'study-id', self.study.id)
        if 'series-id' not in landmarks_data.columns:
            landmarks_data.insert(1, 'series-id', self.__id)

        if data_only:
            return landmarks_data[range(3)].to_numpy().astype(np.float32)
        else:
            return landmarks_data
    
    def has_landmarks(
        self,
        landmark_ids: LandmarkIDs,
        any: bool = False) -> bool:
        real_ids = self.list_landmarks(landmark_ids=landmark_ids)
        req_ids = arg_to_list(landmark_ids, LandmarkID)
        n_overlap = len(np.intersect1d(real_ids, req_ids))
        return n_overlap > 0 if any else n_overlap == len(req_ids)

    def list_landmarks(
        self,
        landmark_ids: LandmarkIDs = 'all') -> List[LandmarkID]:
        # Load landmark IDs.
        landmarks_data = load_csv(self.__filepath)
        ids = list(sorted(landmarks_data['landmark-id']))

        if landmark_ids == 'all':
            return ids

        if isinstance(landmark_ids, float) and landmark_ids > 0 and landmark_ids < 1:
            # Take non-random subset of landmarks.
            ids = p_landmarks(ids, landmark_ids)
        else:
            # Filter based on passed landmarks.
            landmark_ids = arg_to_list(landmark_ids, LandmarkID)
            ids = [i for i in ids if i in landmark_ids]

        return ids

    def __str__(self) -> str:
        return self.__global_id
    