import os
from glob import glob
from tqdm.auto import tqdm

from mymi.utils.varian import process_xim_session

pat_ids = [1]
fractions = [2]
sessions = [1]
basepath = r"L:\LEARN\Orange\Lung\Patient Images"

for p in tqdm(pat_ids, desc='Patients'):
    pat_id = f'PAT{p:02d}'
    for f in tqdm(fractions, desc='Fractions', leave=False):
        for s in tqdm(sessions, desc='Sessions', leave=False):
            # Find all projection directories for this patient/fraction/session.
            pattern = os.path.join(basepath, pat_id, f'Fx{f}', 'CBCT', f'CBCT{s}', 'CBCT Projections', 'IPS', 'Proj *')
            session_paths = sorted(glob(pattern))
            # Exclude already-processed output directories (e.g. 'Proj 0_proc'), which
            # also match the glob pattern above and contain no .xim files to process.
            session_paths = [p for p in session_paths if not p.endswith('_proc')]
            for session_path in tqdm(session_paths, desc='Proj', leave=False):
                process_xim_session(session_path)
