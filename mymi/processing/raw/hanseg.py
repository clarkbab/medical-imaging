import os
from tqdm import tqdm

from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate
from mymi.utils import *

def convert_hanseg_to_nifti() -> None:
    dataset = 'HANSEG'
    rset = RawDataset(dataset)
    nset = recreate(dataset)

    raw_path = os.path.join(rset.path, 'set_1')
    pat_ids = [p for p in os.listdir(raw_path) if p.startswith('case_')]

    # Copy data.
    for p in tqdm(pat_ids):
        # Copy MR/CT.
        rpat_path = os.path.join(raw_path, p)
        filepath = os.path.join(rpat_path, f'{p}_IMG_CT.nrrd')
        data, spacing, origin = load_nrrd(filepath)
        npat_path = os.path.join(nset.path, 'data', 'patients', p)
        filepath = os.path.join(npat_path, 'study_0', 'ct', 'series_0.nii.gz')
        save_nifti(data, filepath, spacing=spacing, origin=origin)
        filepath = os.path.join(raw_path, p, f'{p}_IMG_MR_T1.nrrd')
        data, spacing, origin = load_nrrd(filepath)
        filepath = os.path.join(npat_path, 'study_0', 'mr', 'series_1.nii.gz')
        save_nifti(data, filepath, spacing=spacing, origin=origin)

        # Copy region labels.
        files = [r for r in os.listdir(rpat_path) if r.startswith(f'{p}_OAR_')]
        for f in files:
            filepath = os.path.join(rpat_path, f)
            data, spacing, origin = load_nrrd(filepath)
            region = f.replace(f'{p}_OAR_', '').replace('.seg.nrrd', '')
            filepath = os.path.join(npat_path, 'study_0', 'regions', 'series_2', f'{region}.nii.gz')
            save_nifti(data, filepath, spacing=spacing, origin=origin)
