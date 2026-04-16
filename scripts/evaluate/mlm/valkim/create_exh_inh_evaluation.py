import dicomset as ds
from dicomset.utils.metrics import centroid_error, dice, distances
from dicomset.utils.pandas import append_row
from dicomset.utils.io import save_csv
from dicomset.nifti.utils import load_registered_regions
import pandas as pd
from tqdm import tqdm

# Compare methods.
methods = ['corrfield', 'plastimatch', 'unigradicon', 'unigradicon-io']

cols = {
    'patient-id': str,
    'region': str,
    'method': str,
    'metric': str,
    'value': float,
}
df = pd.DataFrame(columns=cols.keys())

dataset = 'VALKIM-PP'
inh_series = 'series_0'
exh_series = 'series_5'
set = ds.get(dataset, 'nifti')
pat_ids = ['PAT1', 'PAT2', 'PAT3']
regions = ['GTV', 'ts_Lung']

for p in tqdm(pat_ids):
    pat = set.patient(p)
    inh_ct = pat.ct_series(inh_series).data
    inh_affine = pat.ct_series(inh_series).affine
    inh_labels = pat.regions_series(inh_series).data(r=regions)
    exh_ct = pat.ct_series(exh_series).data
    exh_affine = pat.ct_series(exh_series).affine
    exh_labels = pat.regions_series(exh_series).data(r=regions)
    
    # Add moving -> fixed error.
    dices = dice(exh_labels, inh_labels)
    for r, d in zip(regions, dices):
        data = {
            'patient-id': pat.id,
            'region': r,
            'method': 'no-reg',
            'metric': 'dice',
            'value': d,
        }
        df = append_row(df, data)

    cte = centroid_error(exh_labels, inh_labels, affine=inh_affine)
    for r, e in zip(regions, cte):
        data = {
            'patient-id': pat.id,
            'region': r,
            'method': 'no-reg',
            'metric': 'centroid-error',
            'value': e,
        }
        df = append_row(df, data) 

    rdistances = distances(exh_labels, inh_labels, affine=inh_affine)
    for r, d in zip(regions, rdistances):
        for metric, value in d.items():
            data = {
                'patient-id': pat.id,
                'region': r,
                'method': 'no-reg',
                'metric': metric,
                'value': value,
            }
            df = append_row(df, data)    
    
    for m in methods:
        moved_labels, _ = load_registered_regions(dataset, pat.id, m, regions, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
        dices = dice(moved_labels, inh_labels)
        for r, d in zip(regions, dices):
            data = {
                'patient-id': pat.id,
                'region': r,
                'method': m,
                'metric': 'dice',
                'value': d,
            }
            df = append_row(df, data)
        cte = centroid_error(moved_labels, inh_labels, affine=inh_affine)
        for r, e in zip(regions, cte):
            data = {
                'patient-id': pat.id,
                'region': r,
                'method': m,
                'metric': 'centroid-error',
                'value': e,
            }
            df = append_row(df, data) 
        rdistances = distances(moved_labels, inh_labels, affine=inh_affine)
        for r, d in zip(regions, rdistances):
            for metric, value in d.items():
                data = {
                    'patient-id': pat.id,
                    'region': r,
                    'method': m,
                    'metric': metric,
                    'value': value,
                }
                df = append_row(df, data)    

save_csv(df, 'files:mlm/valkim/4dct-inh-exh-eval.csv')
