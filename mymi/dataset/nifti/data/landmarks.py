import numpy as np
import os
from typing import Dict, List, Optional

from mymi.regions import regions_to_list
from mymi.types import Landmarks, PatientLandmark, PatientLandmarks, SeriesID
from mymi.utils import arg_to_list, load_csv, load_nifti

from .data import NiftiData

class LandmarkData(NiftiData):
    def __init__(
        self,
        study: 'NiftiStudy',
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
        landmarks: PatientLandmarks = 'all',
        use_image_coords: bool = False,
        **kwargs) -> Landmarks:

        # Load landmarks.
        lm_df = load_csv(self.__path)
        lm_df = lm_df.rename(columns={ '0': 0, '1': 1, '2': 2 })

        # Filter on landmarks.
        if landmarks != 'all':
            landmarks = regions_to_list(landmarks)
            lm_df = lm_df[lm_df['landmark-id'].isin(landmarks)]

        # Convert to image coordinates.
        if use_image_coords:
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
        landmarks: PatientLandmarks) -> bool:
        landmarks = regions_to_list(landmarks, literals={ 'all': self.list_landmarks })
        pat_landmarks = self.list_landmarks()
        if len(np.intersect1d(landmarks, pat_landmarks)) != 0:
            return True
        else:
            return False

    def list_landmarks(
        self,
        # Only the landmarks in 'landmarks' should be returned.
        # Saves us from performing filtering code elsewhere many times.
        landmarks: Optional[PatientLandmarks] = 'all') -> List[PatientLandmark]:

        # Get landmark names.
        lm_df = self.data(landmarks=landmarks)
        lms = list(lm_df['landmark-id'])

        return lms
    