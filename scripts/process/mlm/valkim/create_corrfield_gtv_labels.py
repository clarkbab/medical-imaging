import dicomset as ds
from dicomset.nifti.utils import create_region, load_registered_regions
from tqdm import tqdm

makeitso = True

dataset = 'VALKIM-PP'
set = ds.get(dataset, 'nifti')
pat_ids = ['PAT1', 'PAT2', 'PAT3']
exh_series = 'series_5'
int_phases = ['series_1', 'series_2', 'series_3', 'series_4', 'series_6', 'series_7', 'series_8', 'series_9']
for p in tqdm(pat_ids):
    for s in tqdm(int_phases, leave=False):
        # Load exh -> int registrations.
        moved_gtv, moved_affine = load_registered_regions(dataset, p, 'corrfield', 'GTV', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)

        # Save moved GTV as new region.
        create_region(dataset, p, 'study_0', s, 'cf_GTV', moved_gtv, affine=moved_affine, makeitso=makeitso)
