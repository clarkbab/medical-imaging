import os
import shutil
from tqdm import tqdm

from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate
from mymi.utils import *

def convert_l2r_lung_ct_to_nifti() -> None:
    dataset = 'L2R-LUNG-CT'
    rset = RawDataset(dataset)
    set = recreate(dataset)

    # Create holdout split file.
    filepath = os.path.join(rset.path, 'training', 'scans')
    pat_ids = [f.split('_')[1] for f in os.listdir(filepath)]
    train_df = pd.DataFrame([pat_ids, ['train'] * len(pat_ids)], columns=['patient-id', 'split'])
    filepath = os.path.join(rset.path, 'test', 'scans')
    pat_ids = [f.split('_')[1] for f in os.listdir(filepath)]
    test_df = pd.DataFrame([pat_ids, ['test'] * len(pat_ids)], columns=['patient-id', 'split'])
    df = pd.concat((train_df, test_df), axis=0)
    filepath = os.path.join(set.path, 'holdout-split.csv')
    save_csv(df, filepath)

    # Copy data.
    folders = ['training', 'test']
    for f in tqdm(folders):
        scanpath = os.path.join(rset.path, f, 'scans')
        lungpath = os.path.join(rset.path, f, 'lungMasks')
        scans = os.listdir(scanpath)
        for s in tqdm(scans, leave=False):
            # Copy CT.
            ctpath = os.path.join(scanpath, s)
            pat_id = s.split('_')[1]
            if s.split('_')[2] == 'insp':
                study_id = 'study_1'
            else:
                study_id = 'study_0'
            destpath = os.path.join(set.path, 'data', 'patients', pat_id, study_id, 'ct', 'series_0.nii.gz')
            os.makedirs(os.path.dirname(destpath), exist_ok=True)
            shutil.copyfile(ctpath, destpath)

            # Copy lung masks.
            lpath = os.path.join(lungpath, s)
            destpath = os.path.join(set.path, 'data', 'patients', pat_id, study_id, 'regions', 'series_1', 'Lung.nii.gz')
            os.makedirs(os.path.dirname(destpath), exist_ok=True)
            shutil.copyfile(lpath, destpath)
