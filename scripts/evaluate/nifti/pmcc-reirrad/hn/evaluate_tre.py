import numpy as np
import pandas as pd
from tqdm import tqdm

from mymi import datasets as ds
from mymi.utils import *

# Load patient.
dataset = 'PMCC-REIRRAD'
models = [
    'rir',
    'dmp',
    'edmp',
    'sg_c',
    'sg_lm',
]
set = ds.get(dataset, 'nifti')
dset = ds.get(dataset, 'dicom')
pat_ids = set.list_patients(group='hn')
dfs = []
for p in tqdm(pat_ids):
    for m in models:
        # set.build_index()
        pat = set.patient(p)
        moving_study = pat.study('i:0')
        fixed_study = pat.study('i:1')
        
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
        dfs.append(df)

df = pd.concat(dfs, axis=0)
tre_df = df[df['metric'] == 'tre']

save_csv(tre_df, 'files:imreg/hn-tre.csv')
