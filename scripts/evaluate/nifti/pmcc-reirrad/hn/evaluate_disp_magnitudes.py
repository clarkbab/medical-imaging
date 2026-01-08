import pandas as pd
from tqdm import tqdm

from mymi import datasets as ds
from mymi.metrics import tre
from mymi.utils import *

# Get rigid model propagated C2 landmarks.
dataset = 'PMCC-REIRRAD'
set = ds.get(dataset, 'nifti')
pat_ids = set.list_patients(group='hn')
models = ['rir', 'dmp', 'edmp', 'sg_c', 'sg_lm']

pat_disp_dfs = []
for p in tqdm(pat_ids):
    # Get RIR-propagated landmarks.
    pat = set.patient(p)
    fixed_study = pat.study('idx:1')
    for i in range(len(models)):
        rir_series = fixed_study.landmarks_series(f'idx:{i + 1}')
        if '/rir.dcm' in rir_series.dicom.filepath:
            rir_lms = rir_series.data()
            break

    disp_dfs = []
    for i in range(len(models)):
        # Get deformably propagated C2 landmarks.
        def_series = fixed_study.landmarks_series(f'idx:{i + 1}')
        # assert f'/{m}.dcm' in def_series.dicom.filepath, def_series.dicom.filepath
        m = def_series.dicom.filepath.split('/')[-1].replace('.dcm', '')
        def_lms = def_series.data()

        # Calculate displacements.
        disps = tre(def_lms, rir_lms)
        disps = disps.rename(columns={ 'tre': 'disp' })
        disps.insert(2, 'model', m)
        disp_dfs.append(disps)
    pat_disp_df = pd.concat(disp_dfs, axis=0)
    pat_disp_dfs.append(pat_disp_df)

disp_df = pd.concat(pat_disp_dfs, axis=0)

# Merge with TRE.
tre_df = load_csv('files:imreg/hn-tre.csv')
tre_df = tre_df.rename(columns={ 'value': 'tre' })
tre_df = pd.merge(tre_df, disp_df, on=['patient-id', 'landmark-id', 'model'], how='left')
tre_df = tre_df[~tre_df['disp'].isna()]

save_csv(tre_df, 'files:imreg/hn-disp-magnitude.csv')
