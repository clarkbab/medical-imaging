import numpy as np
import os
from typing import *

from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

from .data import NiftiData

class LandmarkNiftiData(NiftiData):
    def __init__(
        self,
        study: 'NiftiStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__path = os.path.join(study.path, 'landmarks', f'{id}.csv')
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
        landmarks: Landmarks = 'all',
        use_patient_coords: bool = True,
        **kwargs) -> LandmarkData:

        # Load landmarks.
        lm_df = load_csv(self.__path)
        lm_df = lm_df.rename(columns={ '0': 0, '1': 1, '2': 2 })

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        lm_df = lm_df.sort_values('landmark-id')

        if landmarks != 'all':
            landmarks = self.list_landmarks(landmarks=landmarks)
            lm_df = lm_df[lm_df['landmark-id'].isin(landmarks)]

        # Convert to image coordinates.
        if not use_patient_coords:
            spacing = self.__study.ct_spacing
            offset = self.__study.ct_offset
            lm_data = lm_df[list(range(3))].to_numpy()
            lm_data = (lm_data - offset) / spacing
            lm_data = lm_data.round()
            lm_data = lm_data.astype(np.int32)  # Don't use unsigned int - there could be negative values after registration for example.
            lm_df[list(range(3))] = lm_data

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in lm_df.columns:
            lm_df.insert(0, 'patient-id', self.study.patient.id)
        if 'study-id' not in lm_df.columns:
            lm_df.insert(1, 'study-id', self.study.id)

        return lm_df
    
    # Returns 'True' if has at least one of the passed 'regions'.
    def has_landmarks(
        self,
        landmarks: Landmarks) -> bool:
        landmarks = arg_to_list(landmarks, int, literals={ 'all': self.list_landmarks })
        pat_landmarks = self.list_landmarks()
        if len(np.intersect1d(landmarks, pat_landmarks)) != 0:
            return True
        else:
            return False

    def list_landmarks(
        self,
        # Only the landmarks in 'landmarks' should be returned.
        # Saves us from performing filtering code elsewhere many times.
        landmarks: Optional[Landmarks] = 'all') -> List[Landmark]:

        # Load landmark IDs.
        lm_df = load_csv(self.__path)
        landmark_ids = list(sorted(lm_df['landmark-id']))

        if landmarks == 'all':
            return landmark_ids

        if isinstance(landmarks, float) and landmarks > 0 and landmarks < 1:
            # Take non-random subset of landmarks.
            landmark_ids = p_landmarks(landmark_ids, landmarks)
        else:
            # Filter based on passed landmarks.
            landmarks = arg_to_list(landmarks, int)
            landmark_ids = [i for i in landmark_ids if i in landmarks]

        return landmark_ids
    