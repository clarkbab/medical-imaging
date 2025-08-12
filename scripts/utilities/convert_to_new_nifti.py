import numpy as np
import os
import shutil
import sys

mymi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(mymi_path)

from mymi import datasets as ds

# Convert data.
dataset = 'LUNG-4DCT'
set = ds.get(dataset, 'nifti')
datapath = os.path.join(set.path, 'data')

# Move CT.
ctpath = os.path.join(datapath, 'ct')
files = os.listdir(ctpath)
pat_ids = [f.split('-')[0] for f in files]
pat_ids = list(np.unique(pat_ids))

for p in pat_ids:
    # Move 'moving'.
    src = os.path.join(ctpath, f'{p}-0.nii.gz')
    dest = os.path.join(datapath, 'patients', p, 'study_0', 'ct', 'series_0.nii.gz')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)

    # Move 'fixed'.
    src = os.path.join(ctpath, f'{p}-1.nii.gz')
    dest = os.path.join(datapath, 'patients', p, 'study_1', 'ct', 'series_0.nii.gz')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)

# Move regions.
regionspath = os.path.join(datapath, 'regions')
regions = os.listdir(regionspath)
for p in pat_ids:
    for r in regions:
        # Move 'moving'.
        src = os.path.join(regionspath, r, f'{p}-0.nii.gz')
        dest = os.path.join(datapath, 'patients', p, 'study_0', 'regions', 'series_1', f'{r}.nii.gz')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)

        # Move 'fixed'.
        src = os.path.join(regionspath, r, f'{p}-1.nii.gz')
        dest = os.path.join(datapath, 'patients', p, 'study_1', 'regions', 'series_1', f'{r}.nii.gz')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)

# Move landmarks.
landmarkspath = os.path.join(datapath, 'landmarks')
for p in pat_ids:
    # Move 'moving'.
    src = os.path.join(landmarkspath, f'{p}-0.csv')
    dest = os.path.join(datapath, 'patients', p, 'study_0', 'landmarks', 'series_1.csv')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)

    # Move 'fixed'.
    src = os.path.join(landmarkspath, f'{p}-1.csv')
    dest = os.path.join(datapath, 'patients', p, 'study_1', 'landmarks', 'series_1.csv')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)

# Move registration predictions.
predpath = os.path.join(datapath, 'predictions')
models = os.listdir(predpath)
skip_models = [
    'registration',
    'VOXELMORPH',
]
for p in pat_ids:
    for m in models:
        if m in skip_models:
            continue
        if 'VELOCITY' in m:
            warp_ext = 'tfm'
        else:
            warp_ext = 'hdf5'

        # Move CT.
        src = os.path.join(predpath, m, 'ct', f'{p}-0.nii.gz')
        dest = os.path.join(datapath, 'predictions', 'registration', 'patients', p, 'study_0', p, 'study_1', m, 'ct', 'series_0.nii.gz')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)

        # Move transform.
        src = os.path.join(predpath, m, 'ct', f'{p}-0_warp.{warp_ext}')
        dest = os.path.join(datapath, 'predictions', 'registration', 'patients', p, 'study_0', p, 'study_1', m, 'dvf', f'series_0.{warp_ext}')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)

        # Move landmarks.
        src = os.path.join(predpath, m, 'landmarks', f'{p}-0.csv')
        dest = os.path.join(datapath, 'predictions', 'registration', 'patients', p, 'study_0', p, 'study_1', m, 'landmarks', 'series_1.csv')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
