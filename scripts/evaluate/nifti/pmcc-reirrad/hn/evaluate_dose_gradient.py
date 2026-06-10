import dicomset as ds
from dicomset.utils import load_csv, save_csv, sample, to_numpy, crop_points, logger
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

lm_delta = 1e-3
set = ds.load('PMCC-REIRRAD', 'nifti')
pat_ids = set.list_patients(g='hn')
dset = ds.load('PMCC-REIRRAD', 'dicom')
filepath = os.path.join(dset.path, 'files', 'reg_fovs.csv')
fov_df = load_csv(filepath)
moving_landmarkses = []
for p in tqdm(pat_ids):
    pat = set.patient(p)
    moving_study = pat.study('i:0')
    fixed_study = pat.study('i:1')
    fixed_lm_series = fixed_study.landmarks_series('series_0')
    assert '/C2/' in fixed_lm_series.dicom.filepath, fixed_lm_series.dicom.filepath
    fixed_landmarks = fixed_lm_series.data()
    moving_lm_series = moving_study.landmarks_series('series_0')
    assert '/C1/' in moving_lm_series.dicom.filepath
    moving_landmarks = moving_lm_series.data(sample_dose=True)
    moving_dose_series = moving_study.dose_series('series_0')
    assert '/C1/' in moving_dose_series.dicom.filepath
    moving_dose = moving_dose_series.data

    # Load the lm model FOV.
    pat_info = fov_df[fov_df['patient-id'] == pat.id].iloc[0]
    fov_origin = to_numpy(pat_info[['origin-x', 'origin-y', 'origin-z']].tolist())
    fov_width = to_numpy(pat_info[['fov-width-mm-x', 'fov-width-mm-y', 'fov-width-mm-z']].tolist())
    pat_fov = np.stack([
        fov_origin,
        fov_origin + fov_width,
    ], axis=0)

    # Filter landmarks by FOV.
    tmp_len = len(fixed_landmarks)
    fixed_landmarks = crop_points(fixed_landmarks, pat_fov)
    n_removed = tmp_len - len(fixed_landmarks)
    if n_removed > 0:
        logger.warn(f"Removed {n_removed} fixed landmarks from {p}.")
    moving_landmarks = moving_landmarks[moving_landmarks['landmark-id'].isin(fixed_landmarks['landmark-id'])]
    assert len(moving_landmarks) == len(fixed_landmarks)
    
    # Calculate approx gradient norm at landmarks using finite difference method.
    for a in range(3):
        moving_landmarks_fd = moving_landmarks.copy()
        moving_landmarks_fd[a] = moving_landmarks_fd[a] + lm_delta
        moving_landmarks_fd = sample(moving_dose, moving_landmarks_fd, affine=moving_dose_series.affine, sample_col='dose')
        # moving_df = sample(moving_dose, moving_lms, affine=moving_dose_series.affine)
        moving_landmarks[f'grad-{a}'] = (moving_landmarks_fd['dose'] - moving_landmarks['dose']) / lm_delta
    moving_landmarks['grad-norm'] = np.linalg.norm(moving_landmarks[['grad-0', 'grad-1', 'grad-2']], axis=1)
    moving_landmarkses.append(moving_landmarks)

moving_landmarks = pd.concat(moving_landmarkses, axis=0)

save_csv(moving_landmarks, 'files:imreg/hn-dose-gradient.csv')
