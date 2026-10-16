import os
from tqdm.auto import tqdm

from mymi.utils.elekta import create_his_patient_videos

pat_ids = [1]
fractions = 'all'
# fractions = [0]
basepath = r"L:\LEARN\Nepean\Lung\Patient Images"

for p in tqdm(pat_ids, desc='Patients'):
    pat_path = os.path.join(basepath, f'PAT{p:02d}')
    create_his_patient_videos(pat_path, fraction=fractions)
