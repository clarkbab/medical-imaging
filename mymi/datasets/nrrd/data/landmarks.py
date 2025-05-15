import numpy as np
import os
from typing import Dict, List, Optional

from mymi.regions import regions_to_list
from mymi.typing import Landmarks, Landmark, Landmarks, SeriesID
from mymi.utils import arg_to_list, load_files_csv, load_nrrd

from .data import NrrdData

class LandmarkData(NrrdData):
    def __init__(
        self,
        study: 'NrrdStudy',
        id: SeriesID) -> None:
        self.__id = id
        self.__path = os.path.join(study.path, 'landmarks', f'{id}.csv')
        self.__study = study

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
        **kwargs) -> Landmarks:

        # Load landmarks.
        lm_df = load_files_csv(self.__path)
        lm_df = lm_df.rename(columns={ '0': 0, '1': 1, '2': 2 })

        # Filter on landmarks.
        if landmarks != 'all':
            landmarks = arg_to_list(landmarks, int)
            lm_df = lm_df[lm_df['landmark-id'].isin(landmarks)]

        # Convert to image coordinates.
        if not use_patient_coords:
            spacing = self.__study.ct_spacing
            offset = self.__study.ct_offset
            lm_data = lm_df[list(range(3))]
            lm_data = (lm_data - offset) / spacing
            lm_data = lm_data.round()
            lm_data = lm_data.astype(np.uint32)
            lm_df[list(range(3))] = lm_data

        # Add extra columns - in case we're concatenating landamrks from multiple patients/studies.
        if 'patient-id' not in lm_df.columns:
            lm_df.insert(0, 'patient-id', self.study.patient.id)
        if 'study-id' not in lm_df.columns:
            lm_df.insert(1, 'study-id', self.study.id)

        return lm_df
    
    # Returns 'True' if has at least one of the passed 'regions'.
    def has_landmark(
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

        # Get landmark names.
        lm_df = self.data(landmarks=landmarks)
        lms = list(lm_df['landmark-id'])

        return lms
    