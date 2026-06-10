import dicomset as ds
from dicomset.utils import load_csv, save_csv, sample, to_numpy, crop_points, tre, logger
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Get rigid model propagated C2 landmarks.
dataset = 'PMCC-REIRRAD'
set = ds.load(dataset, 'nifti')
pat_ids = set.list_patients(g='lung')
models = ['rir', 'dmp', 'edmp', 'sg_c', 'sg_lm', 'lm']

dset = ds.load('PMCC-REIRRAD', 'dicom')
filepath = os.path.join(dset.path, 'files', 'reg_fovs.csv')
fov_df = load_csv(filepath)

pat_disp_dfs = []
for p in tqdm(pat_ids):
    # Get RIR-propagated landmarks.
    pat = set.patient(p)
    fixed_study = pat.study('i:1')
    fixed_landmarks = fixed_study.landmarks_series('series_0').data()
    for i in range(len(models)):
        rir_series = fixed_study.landmarks_series(f'i:{i + 1}')
        if '/rir.dcm' in rir_series.dicom.filepath:
            rir_lms = rir_series.data()
            break

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
    print(fixed_landmarks.head())
    n_removed = tmp_len - len(fixed_landmarks)
    if n_removed > 0:
        logger.warn(f"Removed {n_removed} fixed landmarks from {p}.")

    # Filter by removed fixed landmarks.
    rir_lms = rir_lms[rir_lms['landmark-id'].isin(fixed_landmarks['landmark-id'])]
    print(rir_lms.head())

    disp_dfs = []
    for i in range(len(models)):
        # Get deformably propagated C2 landmarks.
        def_series = fixed_study.landmarks_series(f'i:{i + 1}')
        # assert f'/{m}.dcm' in def_series.dicom.filepath, def_series.dicom.filepath
        m = def_series.dicom.filepath.split('/')[-1].replace('.dcm', '')
        def_lms = def_series.data()

        # Filter by removed fixed landmarks.
        def_lms = def_lms[def_lms['landmark-id'].isin(fixed_landmarks['landmark-id'])]
        print(def_lms.head())

        # Calculate displacements.
        disps = tre(def_lms, rir_lms)
        disps = disps.rename(columns={ 'tre': 'disp' })
        disps.insert(2, 'model', m)
        disp_dfs.append(disps)
    pat_disp_df = pd.concat(disp_dfs, axis=0)
    pat_disp_dfs.append(pat_disp_df)

disp_df = pd.concat(pat_disp_dfs, axis=0)

# Merge with TRE.
tre_df = load_csv('files:imreg/lung-tre.csv')
tre_df = tre_df.rename(columns={ 'value': 'tre' })
tre_df = pd.merge(tre_df, disp_df, on=['patient-id', 'landmark-id', 'model'], how='left')
tre_df = tre_df[~tre_df['disp'].isna()]

save_csv(tre_df, 'files:imreg/lung-disp-magnitude.csv')
