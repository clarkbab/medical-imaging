import dicomset as ds
from dicomset.utils import save_csv, logger, crop_points, load_csv, to_numpy
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Load patient.
dataset = 'PMCC-REIRRAD'
models = [
    'rir',
    'dmp',
    'edmp',
    'sg_c',
    'sg_lm',
    'lm',
]
set = ds.load(dataset, 'nifti')
dset = ds.load(dataset, 'dicom')
pat_ids = set.list_patients(g='lung')
dset = ds.load('PMCC-REIRRAD', 'dicom')
filepath = os.path.join(dset.path, 'files', 'reg_fovs.csv')
fov_df = load_csv(filepath)
dfs = []
for p in tqdm(pat_ids):
    # set.build_index()
    pat = set.patient(p)
    moving_study = pat.study('i:0')
    fixed_study = pat.study('i:1')

    # Load the lm model FOV.
    pat_info = fov_df[fov_df['patient-id'] == pat.id].iloc[0]
    fov_origin = to_numpy(pat_info[['origin-x', 'origin-y', 'origin-z']].tolist())
    fov_width = to_numpy(pat_info[['fov-width-mm-x', 'fov-width-mm-y', 'fov-width-mm-z']].tolist())
    pat_fov = np.stack([
        fov_origin,
        fov_origin + fov_width,
    ], axis=0)

    # Get landmarks within the FOV.
    lm_series = fixed_study.landmarks_series('i:0')
    assert '/C2/RTSTRUCT.dcm' in lm_series.dicom.filepath
    lm_df = lm_series.data()
    tmp_len = len(lm_df)
    tmp_landmark_ids = lm_df['landmark-id'].tolist()
    lm_df = crop_points(lm_df, pat_fov)
    n_removed = tmp_len - len(lm_df)
    if n_removed > 0:
        rem_ids = [i for i in tmp_landmark_ids if i not in lm_df['landmark-id'].tolist()]
        logger.warn(f"Removed {n_removed} fixed landmarks from {p}: {rem_ids}")

    for m in models:
        
        # Compare to Velocity TREs.
        filepath = os.path.join(dset.path, 'data', 'velocity', pat.id, f"{m}.txt")
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Convert to dataframe.
        moving_lms = moving_study.landmarks_data()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if l != '' and l != '(mm)']
        lines = np.array(lines)
        
        lines = lines.reshape(-1, 11)
        lines[0, -1] = 'TRE'
        lines[1:, 0] = moving_lms['landmark-id'].values
        df = pd.DataFrame(lines[1:], columns=lines[0])
        df = df.astype({ 'TRE': float })
        df = df.rename(columns={ 'Point Name': 'landmark-id', 'TRE': 'value' })
        df.insert(0, 'patient-id', p)
        df.insert(2, 'model', m)
        df.insert(3, 'metric', 'tre')

        # Filter by landmarks in the fov.
        tmp_len = len(df)
        df = df[df['landmark-id'].isin(lm_df['landmark-id'])]
        n_removed = tmp_len - len(df)
        if n_removed > 0:
            logger.warn(f"Removed {n_removed} landmarks from {p} {m}.")

        dfs.append(df)

df = pd.concat(dfs, axis=0)
tre_df = df[df['metric'] == 'tre']

save_csv(tre_df, 'files:imreg/lung-tre.csv')
