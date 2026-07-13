import os
from tqdm.auto import tqdm

from mymi.utils.cdog import create_patient_shrouds

pat_ids = [1, 2, 3, 4]
pat_ids = [4]
fractions = 'all'
arcs = 'all'
pathpath = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"

for i in tqdm(pat_ids, desc='Patients'):
    pat_id = f'Patient{i:02d}'
    pat_path = os.path.join(pathpath, pat_id)
    create_patient_shrouds(pat_path, fraction=fractions, arc=arcs)
