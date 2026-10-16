import os
from tqdm.auto import tqdm

from mymi.utils.elekta import process_his_fraction

pat_ids = [1]
fractions = [0]
basepath = r"L:\LEARN\Nepean\Lung\Patient Images"

for p in tqdm(pat_ids, desc='Patients'):
    pat_id = f'PAT{p:02d}'
    for f in tqdm(fractions, desc='Fractions', leave=False):
        fracpath = os.path.join(basepath, pat_id, f'FX{f}')
        process_his_fraction(fracpath)
