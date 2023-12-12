import os
import sys
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.dataset import NIFTIDataset
from mymi.registration.dataset.nifti import create_patient_registration

dataset = 'PMCC-HN-REPLAN'
set = NIFTIDataset(dataset)
pat_ids = set.list_patients()
n_pats = len(pat_ids) // 2

for i in tqdm(range(n_pats)):
    fixed_pat_id = f'{i}-1'
    moving_pat_id = f'{i}-0'
    create_patient_registration(dataset, fixed_pat_id, moving_pat_id)
