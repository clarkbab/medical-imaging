import numpy as np
import os
from typing import *

from mymi.transforms import sample
from mymi.typing import *
from mymi.utils import *

from .series import NiftiSeries
from .images import CtImageSeries, DoseImageSeries

class LandmarksSeries(NiftiSeries):
    def __init__(
        self,
        dataset_id: DatasetID,
        pat_id: PatientID,
        study_id: StudyID,
        id: NiftiSeriesID,
        filepath: FilePath,
        ref_ct: Optional[CtImageSeries] = None,
        ref_dose: Optional[DoseImageSeries] = None) -> None:
        self.__dataset_id = dataset_id
        self.__filepath = filepath
        self._global_id = f'NIFTI:{dataset_id}:{pat_id}:{study_id}:{id}'
        self.__id = id
        self.__pat_id = pat_id
        self.__ref_ct = ref_ct
        self.__ref_dose = ref_dose
        self.__study_id = study_id

    def data(
        self,
        data_only: bool = False,
        landmark_ids: LandmarkIDs = 'all',
        sample_ct: bool = False,
        sample_dose: bool = False,
        use_patient_coords: bool = True,
        **kwargs) -> Union[LandmarksData, LandmarksVoxelData, Points3D, Voxels]:

        # Load landmarks.
        landmark_data = load_csv(self.__filepath)
        landmark_data = landmark_data.rename(columns={ '0': 0, '1': 1, '2': 2 })
        if not use_patient_coords:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot convert landmarks to image coordinates without 'ref_ct'.")
            landmark_data = landmarks_to_image_coords(landmark_data, self.__ref_ct.spacing, self.__ref_ct.offset)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        landmark_data = landmark_data.sort_values('landmark-id')

        if landmark_ids != 'all':
            landmark_ids = self.list_landmarks(landmark_ids=landmark_ids)
            landmark_data = landmark_data[landmark_data['landmark-id'].isin(landmark_ids)]

        # Add sampled CT intensities.
        if sample_ct:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot sample CT intensities without 'ref_ct'.")
            ct_values = sample(self.__ref_ct.data, landmarks_to_data(landmark_data), spacing=self.__ref_ct.spacing, offset=self.__ref_ct.offset, **kwargs)
            landmark_data['ct-data-id'] = self.__ref_ct.id
            landmark_data['ct'] = ct_values

        # Add sampled dose intensities.
        if sample_dose:
            if self.__ref_dose is None:
                raise ValueError(f"Cannot sample dose intensities without 'ref_dose'.")
            dose_values = sample(self.__ref_dose.data, landmarks_to_data(landmark_data), spacing=self.__ref_dose.spacing, offset=self.__ref_dose.offset, **kwargs)
            landmark_data['dose-data-id'] = self.__ref_dose.id
            landmark_data['dose'] = dose_values

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in landmark_data.columns:
            landmark_data.insert(0, 'patient-id', self.__pat_id)
        if 'study-id' not in landmark_data.columns:
            landmark_data.insert(1, 'study-id', self.__study_id)
        if 'data-id' not in landmark_data.columns:
            landmark_data.insert(2, 'data-id', self.__id)

        if data_only:
            return landmark_data[range(3)].to_numpy().astype(np.float32)
        else:
            return landmark_data

    def has_landmarks(
        self,
        landmark_ids: LandmarkIDs = 'all',
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmark_ids = arg_to_list(landmark_ids, LandmarkID, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(landmark_ids, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmark_ids)

    def list_landmarks(
        self,
        landmark_ids: LandmarkIDs = 'all') -> List[LandmarkID]:
        # Load landmark IDs.
        landmark_data = load_csv(self.__filepath)
        ids = list(sorted(landmark_data['landmark-id']))

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
    