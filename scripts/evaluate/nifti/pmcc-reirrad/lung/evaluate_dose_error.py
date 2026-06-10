import dicomset as ds
from dicomset.utils import load_csv, save_csv, sample, to_numpy, crop_points, logger
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
dose_dfs = []
for p in tqdm(pat_ids):
    # if p != 'PMCC_ReIrrad_L27':
    #     continue
    pat = set.patient(p)
    fixed_study = pat.study('i:1')
    moving_study = pat.study('i:0')

    # Load the lm model FOV.
    pat_info = fov_df[fov_df['patient-id'] == pat.id].iloc[0]
    fov_origin = to_numpy(pat_info[['origin-x', 'origin-y', 'origin-z']].tolist())
    fov_width = to_numpy(pat_info[['fov-width-mm-x', 'fov-width-mm-y', 'fov-width-mm-z']].tolist())
    pat_fov = np.stack([
        fov_origin,
        fov_origin + fov_width,
    ], axis=0)

    for m in models:
        # Get model/dose series.
        model_series = {}
        for s in fixed_study.list_dose_series():
            series = fixed_study.dose_series(s)
            if f'C2_PROP/RTDOSE/{m}.dcm' in series.dicom.filepath:
                model_series[m] = s

        # Sample moving dose at moving landmarks and moved dose at fixed landmarks - calculate difference.
        moving_study = pat.study('i:0')
        moving_lm_series = moving_study.landmarks_series('series_0')
        assert '/C1/' in moving_lm_series.dicom.filepath, moving_lm_series.dicom.filepath
        moving_lms = moving_lm_series.data()
        moving_dose_series = moving_study.dose_series('series_0')
        assert '/C1/' in moving_dose_series.dicom.filepath, moving_dose_series.dicom.filepath
        moving_dose = moving_dose_series.data
        
        fixed_lm_series = fixed_study.landmarks_series('series_0')
        assert '/C2/' in fixed_lm_series.dicom.filepath, fixed_lm_series.dicom.filepath
        print(fixed_lm_series)
        fixed_lms = fixed_lm_series.data()
        moved_dose_series = fixed_study.dose_series(model_series[m])
        assert f'/C2_PROP/RTDOSE/{m}.dcm' in moved_dose_series.dicom.filepath, moved_dose_series.dicom.filepath
        moved_dose = moved_dose_series.data

        # Filter landmarks by FOV.
        tmp_len = len(fixed_lms)
        print(pat_fov)
        print(fixed_lms)
        fixed_lms = crop_points(fixed_lms, pat_fov)
        print(fixed_lms)
        n_removed = tmp_len - len(fixed_lms)
        if n_removed > 0:
            logger.warn(f"Removed {n_removed} fixed landmarks from {p} {m}.")
        moving_lms = moving_lms[moving_lms['landmark-id'].isin(fixed_lms['landmark-id'])]
        assert len(moving_lms) == len(fixed_lms)

        moving_df = sample(moving_dose, moving_lms, affine=moving_dose_series.affine)
        moved_df = sample(moved_dose, fixed_lms, affine=moved_dose_series.affine)

        moved_df['diff'] = moved_df['sample'] - moving_df['sample']
        moved_df['diff-abs'] = moved_df['diff'].abs()
        moved_df.insert(3, 'model', m)
        dose_dfs.append(moved_df)

dose_df = pd.concat(dose_dfs, axis=0)
dose_df.insert(4, 'metric', 'dose-error')
dose_df = dose_df.rename(columns={ 'diff-abs': 'value' })

save_csv(dose_df, 'files:imreg/lung-dose.csv')
