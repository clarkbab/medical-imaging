import os
import subprocess
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi import logging
from mymi.typing import *

def create_corrfield_landmarks(
    dataset: str,
    fixed_region: Region = 'Lung',
    fixed_study_id: StudyID = 'study_1',
    moving_study_id: StudyID = 'study_0',
    pat_ids: PatientIDs = 'all',
    save_as_labels: bool = False,
    splits: Splits = 'all') -> None:

    # Load patient IDs.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids, splits=splits)

    for p in tqdm(pat_ids)[:1]:
        pat = set.patient(p)
        fixed_study = pat.study(fixed_study_id)
        moving_study = pat.study(moving_study_id)

        # Make total seg prediction.
        if save_as_labels:
            output_path = os.path.join(set.path, 'data', 'patients', p, s, 'regions', 'series_1')
        else:
            pred_base = os.path.join(set.path, 'data', 'predictions', 'registration', p, fixed_study_id, p, moving_study_id)

        # Save keypoint correspondences.
        corr_path = os.path.join(pred_base, 'corr.csv')
        command = [
            'corrfield',
            '-F', fixed_study.ct_path,
            '-M', moving_study.ct_path,
            '-m', fixed_study.regions_path(fixed_region),
            '-O', corr_path,
        ]
        logging.info(command)
        subprocess.run(command)
