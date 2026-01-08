import numpy as np
import pandas as pd
from tqdm import tqdm

from mymi import datasets as ds
from mymi.transforms import sample
from mymi.utils import *

lm_delta = 1e-3
set = ds.get('PMCC-REIRRAD', 'nifti')
pat_ids = set.list_patients(group='hn')
moving_landmarkses = []
for p in tqdm(pat_ids):
    pat = set.patient(p)
    moving_study = pat.study('idx:0')
    moving_lm_series = moving_study.landmarks_series('series_1')
    assert '/C1/' in moving_lm_series.dicom.filepath
    moving_landmarks = moving_lm_series.data(sample_dose=True)
    moving_dose_series = moving_study.dose_series('series_2')
    assert '/C1/' in moving_dose_series.dicom.filepath
    moving_dose = moving_dose_series.data
    
    # Calculate approx gradient norm at landmarks using finite difference method.
    for a in range(3):
        moving_landmarks_fd = moving_landmarks.copy()
        moving_landmarks_fd[a] = moving_landmarks_fd[a] + lm_delta
        moving_landmarks_fd = sample(moving_dose, moving_landmarks_fd, spacing=moving_dose_series.spacing, origin=moving_dose_series.origin, landmarks_col='dose')
        moving_landmarks[f'grad-{a}'] = (moving_landmarks_fd['dose'] - moving_landmarks['dose']) / lm_delta
    moving_landmarks['grad-norm'] = np.linalg.norm(moving_landmarks[['grad-0', 'grad-1', 'grad-2']], axis=1)
    moving_landmarkses.append(moving_landmarks)

moving_landmarks = pd.concat(moving_landmarkses, axis=0)

save_csv(moving_landmarks, 'files:imreg/hn-dose-gradient.csv')
