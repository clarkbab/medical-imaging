import numpy as np
import os
from typing import *

from mymi.transforms import sample
from mymi.typing import *
from mymi.utils import *

from ...dicom import DicomDataset, DicomRtStructSeries
from .images import NiftiCtSeries, NiftiDoseSeries
from .series import NiftiSeries

class NiftiLandmarksSeries(NiftiSeries):
    def __init__(
        self,
        dataset: DatasetID,
        pat: PatientID,
        study: StudyID,
        id: NiftiSeriesID,
        index: Optional[pd.DataFrame] = None,
        ref_ct: Optional[NiftiCtSeries] = None,
        ref_dose: Optional[NiftiDoseSeries] = None) -> None:
        super().__init__('landmarks', dataset, pat, study, id, index=index)
        self.__filepath = os.path.join(config.directories.datasets, 'nifti', self._dataset_id, 'data', 'patients', self._pat_id, self._study_id, self._modality, f'{self._id}.csv')
        if not os.path.exists(self.__filepath):
            raise ValueError(f"No NiftiLandmarksSeries '{self._id}' found for study '{self._study_id}'. Filepath: {self.__filepath}")
        self.__ref_ct = ref_ct
        self.__ref_dose = ref_dose

    def data(
        self,
        points_only: bool = False,
        landmark: LandmarkIDs = 'all',
        n: Optional[int] = None,
        sample_ct: bool = False,
        sample_dose: bool = False,
        use_patient_coords: bool = True,
        **kwargs) -> Union[LandmarksFrame, LandmarksFrameVox, Points3D, Voxels]:

        # Load landmarks.
        landmarks_data = load_csv(self.__filepath)
        landmarks_data = landmarks_data.rename(columns={ '0': 0, '1': 1, '2': 2 })
        if not use_patient_coords:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot convert landmarks to image coordinates without 'ref_ct'.")
            landmarks_data = landmarks_to_image_coords(landmarks_data, self.__ref_ct.spacing, self.__ref_ct.origin)

        # Sort by landmark IDs - this means that 'n_landmarks' will be consistent between
        # Dicom/Nifti dataset types.
        landmarks_data = landmarks_data.sort_values('landmark-id')

        # Filter by landmark ID.
        if landmark != 'all':
            landmarks = self.list_landmarks(landmark=landmark)
            landmarks_data = landmarks_data[landmarks_data['landmark-id'].isin(landmarks)]

        # Filter by number of rows.
        if n is not None:
            landmarks_data = landmarks_data.iloc[:n]

        # Add sampled CT intensities.
        if sample_ct:
            if self.__ref_ct is None:
                raise ValueError(f"Cannot sample CT intensities without 'ref_ct'.")
            ct_values = sample(self.__ref_ct.data, landmarks_to_data(landmarks_data), spacing=self.__ref_ct.spacing, origin=self.__ref_ct.origin, **kwargs)
            landmarks_data['ct-series-id'] = self.__ref_ct.id
            landmarks_data['ct'] = ct_values

        # Add sampled dose intensities.
        if sample_dose:
            if self.__ref_dose is None:
                raise ValueError(f"Cannot sample dose intensities without 'ref_dose'.")
            dose_values = sample(self.__ref_dose.data, landmarks_to_data(landmarks_data), spacing=self.__ref_dose.spacing, origin=self.__ref_dose.origin, **kwargs)
            landmarks_data['dose-series-id'] = self.__ref_dose.id
            landmarks_data['dose'] = dose_values

        # Add extra columns - in case we're concatenating landmarks from multiple patients/studies.
        if 'patient-id' not in landmarks_data.columns:
            landmarks_data.insert(0, 'patient-id', self._pat_id)
        if 'study-id' not in landmarks_data.columns:
            landmarks_data.insert(1, 'study-id', self._study_id)
        if 'series-id' not in landmarks_data.columns:
            landmarks_data.insert(2, 'series-id', self._id)

        if points_only:
            return landmarks_data[range(3)].to_numpy().astype(np.float32)
        else:
            return landmarks_data

    @property
    def dicom(self) -> DicomRtStructSeries:
        if self._index is None:
            raise ValueError(f"Dataset did not originate from dicom (no 'index.csv').")
        index = self._index[['dataset', 'patient-id', 'study-id', 'series-id', 'modality', 'dicom-dataset', 'dicom-patient-id', 'dicom-study-id', 'dicom-series-id']]
        index = index[(index['dataset'] == self._dataset_id) & (index['patient-id'] == self._pat_id) & (index['study-id'] == self._study_id) & (index['series-id'] == self._id) & (index['modality'] == 'landmarks')].drop_duplicates()
        assert len(index) == 1
        row = index.iloc[0]
        return DicomDataset(row['dicom-dataset']).patient(row['dicom-patient-id']).study(row['dicom-study-id']).rtstruct_series(row['dicom-series-id'])

    def has_landmark(
        self,
        landmarks: LandmarkIDs,
        any: bool = False,
        **kwargs) -> bool:
        all_ids = self.list_landmarks(**kwargs)
        landmarks = arg_to_list(landmarks, LandmarkID, literals={ 'all': all_ids })
        n_overlap = len(np.intersect1d(landmarks, all_ids))
        return n_overlap > 0 if any else n_overlap == len(landmarks)

    def list_landmarks(
        self,
        landmark: LandmarkIDs = 'all') -> List[LandmarkID]:
        # Load landmark IDs.
        landmarks_data = load_csv(self.__filepath)
        ids = list(sorted(landmarks_data['landmark-id']))

        if landmark == 'all':
            return ids

        if isinstance(landmark, float) and landmark > 0 and landmark < 1:
            # Take non-random subset of landmarks.
            ids = p_landmarks(ids, landmark)
        else:
            # Filter based on passed landmarks.
            landmarks = arg_to_list(landmark, LandmarkID)
            ids = [i for i in ids if i in landmarks]

        return ids

    def __str__(self) -> str:
        return super().__str__(self.__class__.__name__)
    
# Add properties.
props = ['filepath']
for p in props:
    setattr(NiftiLandmarksSeries, p, property(lambda self, p=p: getattr(self, f'_{NiftiLandmarksSeries.__name__}__{p}')))
