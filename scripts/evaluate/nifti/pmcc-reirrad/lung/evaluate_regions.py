import dicomset as ds
from dicomset.utils import dice, distances, save_csv, append_row, region_to_list, load_csv, to_numpy, crop
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Load fixed regions.
dataset = 'PMCC-REIRRAD'
models = ['velocity-dmp', 'velocity-edmp', 'velocity-rir', 'velocity-sg_c', 'velocity-sg_lm', 'velocity-lm']
set = ds.load(dataset, 'nifti')
pat_ids = set.list_patients(g='lung')
cols = {
    'patient-id': str,
    'region': str,
    'model': str,
    'metric': str,
    'value': float,
}
res_df = pd.DataFrame(columns=cols.keys())
dset = ds.load('PMCC-REIRRAD', 'dicom')
filepath = os.path.join(dset.path, 'files', 'reg_fovs.csv')
fov_df = load_csv(filepath)
for p in tqdm(pat_ids):
    pat = set.patient(p)
    moving_study = pat.study('i:0')
    fixed_study = pat.study('i:1')
    fixed_affine = fixed_study.ct_affine
    fixed_regions_series = fixed_study.regions_series('series_0')
    assert '/C2/' in fixed_regions_series.dicom.filepath, fixed_regions_series.dicom.filepath
    fixed_regions, fixed_regions_data = fixed_regions_series.data(r='l:lung')

    # Get series IDs for moved regions.
    model_series = {}
    for s in fixed_study.list_regions_series():
        series = fixed_study.regions_series(s)
        for m in models:
            if f"/{m.split('-')[1]}.dcm" in series.dicom.filepath:
                model_series[m] = series

    # Load the lm model FOV.
    pat_info = fov_df[fov_df['patient-id'] == pat.id].iloc[0]
    fov_origin = to_numpy(pat_info[['origin-x', 'origin-y', 'origin-z']].tolist())
    fov_width = to_numpy(pat_info[['fov-width-mm-x', 'fov-width-mm-y', 'fov-width-mm-z']].tolist())
    pat_fov = np.stack([
        fov_origin,
        fov_origin + fov_width,
    ], axis=0)

    # Evaluate moved regions for each model.
    regions = region_to_list('l:lung', region_map=set.region_map)
    for m in tqdm(models, leave=False):
        moved_regions, moved_regions_data = model_series[m].data(r='l:lung')
        for r in regions:
            if r in fixed_regions and r in moved_regions:
                fdata = fixed_regions_data[fixed_regions.index(r)]
                mdata = moved_regions_data[moved_regions.index(r)]
                print(fdata.shape, fdata.sum())
                print(mdata.shape, mdata.sum())

                fdata = crop(fdata, pat_fov, affine=fixed_affine)
                mdata = crop(mdata, pat_fov, affine=fixed_affine)


                dice_val = dice(mdata, fdata)
                data = {
                    'patient-id': pat.id,
                    'region': r,
                    'model': m,
                    'metric': 'dice',
                    'value': dice_val,
                }
                res_df = append_row(res_df, data)
    
                dists = distances(mdata, fdata, affine=fixed_affine, tol=[1])
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
