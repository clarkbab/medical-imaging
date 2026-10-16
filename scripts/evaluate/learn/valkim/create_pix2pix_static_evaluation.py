import dicomset as ds
from dicomset import config
from dicomset.utils import arg_to_list
from dicomset.utils.io import save_csv
from dicomset.utils.metrics import centroid_error, dice, distances
from dicomset.utils.pandas import append_row
from dicomset.nifti.utils import load_dataset as load_nifti_dataset
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from mymi.utils.cdog import list_tiff_arcs, list_tiff_fractions, load_tiff
from mymi.utils.projections import affine_at_isocentre

project = 'MLM-VALKIM'
model_base = 'pix2pix2d-static'
model_v = 9
# epochs = [epoch]
epochs = list(range(100))

tdataset = 'VALKIM-PP-STATIC'
tset = ds.load(tdataset, 'training')
pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1, 2, 3]
arcs = [1]

cols = {
    'epoch': int,
    'patient-id': str,
    'fraction': int,
    'arc': int,
    'frame': int,
    'kv-source-angle': float,
    'metric': str,
    'value': float,
}

nifti_info = load_nifti_dataset('VALKIM-PP').params['patient-info']

for epoch in tqdm(epochs, desc='Epochs'):
    for p in tqdm(pat_ids, leave=False):
        model_name = f'{model_base}-PAT{p}-{model_v:03d}'

        # Build 2D affine from detector spacing for physical-space metrics (mm).
        # This needs to be scaled to spacing at isocentre - approximately where
        # the GTV is.
        filepath = nifti_info[f'PAT{p}']['treatment-image']
        _, tiff_info = load_tiff(filepath)
        sx, sy = tiff_info['det-spacing']
        affine = np.diag([sx, sy, 1.0])
        affine = affine_at_isocentre(affine, sid=tiff_info['sid'], sdd=tiff_info['sdd'])

        patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
        pat_fractions = arg_to_list(fractions, int, literals={'all': list_tiff_fractions(patpath)})

        for f in tqdm(pat_fractions, leave=False):
            fracpath = os.path.join(patpath, f"Fx{f:02d}")
            pat_arcs = arg_to_list(arcs, int, literals={'all': list_tiff_arcs(fracpath)})

            for a in tqdm(pat_arcs, leave=False):
                df = pd.DataFrame(columns=cols.keys())
                testdir = os.path.join(tset.path, 'data', 'test', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
                preddir = os.path.join(tset.path, 'data', 'predictions', model_name, f'epoch_{epoch}', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')

                pred_files = sorted(fn for fn in os.listdir(preddir) if fn.endswith('_pred.npy'))

                for fname in tqdm(pred_files, desc='Evaluating', leave=False):
                    basename = fname.replace('_pred.npy', '')
                    frame, kv_source_angle = int(basename.split('_')[0]), float(basename.split('_')[1])

                    pred = np.load(os.path.join(preddir, fname)).astype(np.bool_)      # (H, W)
                    label = np.load(os.path.join(testdir, f'{basename}_label.npy')).astype(np.bool_)  # (H, W)

                    row_base = {
                        'epoch': epoch,
                        'patient-id': f'PAT{p}',
                        'fraction': f,
                        'arc': a,
                        'frame': frame,
                        'kv-source-angle': kv_source_angle,
                    }

                    d = dice(pred, label)
                    df = append_row(df, {**row_base, 'metric': 'dice', 'value': d})

                    if pred.sum() > 0 and label.sum() > 0:
                        ce = centroid_error(pred, label, affine=affine)
                        df = append_row(df, {**row_base, 'metric': 'centroid-error', 'value': ce})
                    else:
                        df = append_row(df, {**row_base, 'metric': 'centroid-error', 'value': np.nan})

                    if pred.sum() > 0 and label.sum() > 0:
                        dists = distances(pred, label, affine=affine)
                        for metric, value in dists.items():
                            df = append_row(df, {**row_base, 'metric': metric, 'value': value})
                    else:
                        for metric in ['msd', 'hd95']:
                            df = append_row(df, {**row_base, 'metric': metric, 'value': np.nan})

                evalpath = os.path.join(tset.path, 'data', 'evaluation', model_name, f'epoch_{epoch}', f'PAT{p}', f'Fx{f:02d}', f'Arc{a:02d}')
                os.makedirs(evalpath, exist_ok=True)
                save_csv(df.astype(cols), os.path.join(evalpath, 'metrics.csv'))
                print(f"Saved to {evalpath}")
