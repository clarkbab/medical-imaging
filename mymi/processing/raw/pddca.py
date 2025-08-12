import os
import shutil
from tqdm import tqdm

from mymi.datasets import RawDataset
from mymi.datasets.nifti import recreate as recreate_nifti

def convert_pddca_to_nifti() -> None:
    dataset = 'PDDCA'
    set = RawDataset(dataset)
    dset = recreate_nifti(dataset)
    patpath = os.path.join(set.path, 'data')
    pat_ids = os.listdir(patpath)
    for p in tqdm(pat_ids):
        # Copy CT.
        srcpath = os.path.join(patpath, p, 'img.nrrd')
        studypath = os.path.join(dset.path, 'data', 'patients', p, 'study_0')
        destpath = os.path.join(studypath, 'ct', 'series_0.nrrd')
        os.makedirs(os.path.dirname(destpath), exist_ok=True)
        shutil.copy(srcpath, destpath)

        # Copy regions.
        for r in os.listdir(os.path.join(patpath, p, 'structures')):
            srcpath = os.path.join(patpath, p, 'structures', r)
            destpath = os.path.join(studypath, 'regions', 'series_1', r)
            os.makedirs(os.path.dirname(destpath), exist_ok=True)
            shutil.copy(srcpath, destpath)
