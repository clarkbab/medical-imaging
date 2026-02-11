import pandas as pd
from tqdm import tqdm

from mymi import datasets as ds
from mymi.metrics import dice, distances
from mymi.regions import *
from mymi.utils import *

# Load fixed regions.
dataset = 'PMCC-REIRRAD'
models = ['velocity-dmp', 'velocity-edmp', 'velocity-rir', 'velocity-sg_c', 'velocity-sg_lm']
set = ds.get(dataset, 'nifti')
pat_ids = set.list_patients(group='lung')
cols = {
    'patient-id': str,
    'region': str,
    'model': str,
    'metric': str,
    'value': float,
}
res_df = pd.DataFrame(columns=cols.keys())
for p in tqdm(pat_ids):
    pat = set.patient(p)
    moving_study = pat.study('i:0')
    fixed_study = pat.study('i:1')
    fixed_spacing = fixed_study.ct_spacing
    fixed_regions_series = fixed_study.regions_series('series_1')
    assert '/C2/' in fixed_regions_series.dicom.filepath, fixed_regions_series.dicom.filepath
    fixed_regions = fixed_regions_series.data(region='rl:pmcc-reirrad-lung')

    # Get series IDs for moved regions.
    model_series = {}
    for s in fixed_study.list_regions_series():
        series = fixed_study.regions_series(s)
        for m in models:
            if f"/{m.split('-')[1]}.dcm" in series.dicom.filepath:
                model_series[m] = series

    # Evaluate moved regions for each model.
    regions = regions_to_list('rl:pmcc-reirrad-lung')
    for m in models:
        moved_regions = model_series[m].data(region='rl:pmcc-reirrad-lung')
        for r in regions:
            if r in fixed_regions and r in moved_regions:
                fdata = fixed_regions[r]
                mdata = moved_regions[r]
                dice_val = dice(mdata, fdata)
                data = {
                    'patient-id': pat.id,
                    'region': r,
                    'model': m,
                    'metric': 'dice',
                    'value': dice_val,
                }
                res_df = append_row(res_df, data)
    
                dists = distances(mdata, fdata, fixed_spacing, tol=[1])
                for met, v in dists.items():
                    data = {
                        'patient-id': pat.id,
                        'region': r,
                        'model': m,
                        'metric': met,
                        'value': v,
                    }
                    res_df = append_row(res_df, data)

save_csv(res_df, 'files:imreg/lung-seg.csv')
