import os
from tqdm.auto import tqdm

from mymi.utils.cdog import process_tiff_patient

# Patient01: filename angle is kv-source, scale is IEC61217.

pat_ids = [1, 2, 3, 4]
filename_angles = ['kv-source', 'kv-source', None, None]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]
pathpath = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"

for i in tqdm(pat_ids, desc='Patients'):
    pat_id = f'Patient{i:02d}'
    pat_path = os.path.join(pathpath, pat_id)
    process_tiff_patient(pat_path, fraction=fractions, arc=arcs, filename_angle=filename_angles[i])

    break
