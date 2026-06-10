import os
from tqdm.auto import tqdm

from mymi.utils.cdog import create_tiff_patient_videos

pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]
projections = True
proj_method = 'interp'
n_frames = None
pathpath = r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files"

for i in tqdm(pat_ids, desc='Patients'):
    pat_id = f'Patient{i:02d}'
    pat_path = os.path.join(pathpath, pat_id)
    create_tiff_patient_videos(pat_path, projections=projections, proj_method=proj_method, fraction=fractions, arc=arcs, n_frames=n_frames)
