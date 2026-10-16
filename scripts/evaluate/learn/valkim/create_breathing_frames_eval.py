from datetime import datetime
import dicomset as ds
import magic
import numpy as np
import os
import pandas as pd
import pydicom as dcm
from dicomset.utils import plot_slice, append_row, to_numpy, save_csv

from mymi.utils.cdog import load_tiff_arc, load_tiff_arc_shroud, plot_tiff_arc_shroud, list_tiff_fractions, list_tiff_arcs

def circular_min_distance(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)

patients = [1, 2]
fractions = 'all'
arcs = 'all'

cols = {
    'patient-id': str,
    'fraction': str,
    'arc': int,
    'metric': str,
    'value': float,
}
df = pd.DataFrame(columns=cols.keys())

for p in patients:
    print('patient: ', p)
    f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
    if fractions == 'all':
        pat_fractions = list_tiff_fractions(f)
    else:
        pat_fractions = fractions

    for fx in pat_fractions:
        print('fraction: ', fx)
        # Plot CBCT arc.
        f = fr"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}\Fx{fx:02d}"
        if arcs == 'all':
            pat_arcs = list_tiff_arcs(f)
        else:
            pat_arcs = arcs
        
        for a in pat_arcs:
            print('arc: ', a)
            data, info = load_tiff_arc(f, a)
            shroud, _ = load_tiff_arc_shroud(f, a)
            
            # Get max inhale frames.
            frames = []
            for i in range(len(info)):
                if i in (0, len(info) - 1):
                    continue
                if info.iloc[i]['MMPhase0'] < info.iloc[i - 1]['MMPhase0'] and info.iloc[i]['MMPhase0'] < info.iloc[i + 1]['MMPhase0']:
                    frames.append(i)
            
            # How many frames per breath?
            breath_frames = np.diff(frames)
            bf_mean, bf_std, bf_min, bf_max = float(breath_frames.mean()), float(breath_frames.std()), float(breath_frames.min()), float(breath_frames.max())
            base_data = {
                'patient-id': p,
                'fraction': fx,
                'arc': a,
            }
            metrics = ['fpb-mean', 'fpb-min', 'fpb-max', 'fpb-std']
            values = [bf_mean, bf_min, bf_max, bf_std]
            for m, v in zip(metrics, values):
                data = {
                    **base_data, 
                    'metric': m,
                    'value': v,
                }
                df = append_row(df, data)

            # What's the gantry rotation per breath?
            kv_dists = []
            for i in range(1, len(frames)):
                dist = circular_min_distance(info.iloc[frames[i]]['kv-source-angle'], info.iloc[frames[i - 1]]['kv-source-angle'])
                kv_dists.append(dist)
            kv_dists = to_numpy(kv_dists)

            kvd_mean, kvd_std, kvd_min, kvd_max = float(kv_dists.mean()), float(kv_dists.std()), float(kv_dists.min()), float(kv_dists.max())
            base_data = {
                'patient-id': p,
                'fraction': fx,
                'arc': a,
            }
            metrics = ['kvd-mean', 'kvd-min', 'kvd-max', 'kvd-std']
            values = [kvd_mean, kvd_min, kvd_max, kvd_std]
            for m, v in zip(metrics, values):
                data = {
                    **base_data, 
                    'metric': m,
                    'value': v,
                }
                df = append_row(df, data)

            # Add time stats.
            times = info.iloc[frames][['StartTime', 'StopTime']].values
            elapsed_times = []
            for i in range(1, len(times)):
                t0 = datetime.strptime(times[i - 1][0], '%Y-%m-%d %H:%M:%S.%f')
                t1 = datetime.strptime(times[i][0], '%Y-%m-%d %H:%M:%S.%f')
                diff = t1 - t0
                elapsed_times.append(diff.seconds)
            elapsed_times = to_numpy(elapsed_times)
            
            et_mean, et_std, et_min, et_max = float(elapsed_times.mean()), float(elapsed_times.std()), float(elapsed_times.min()), float(elapsed_times.max())
            base_data = {
                'patient-id': p,
                'fraction': fx,
                'arc': a,
            }
            metrics = ['et-mean', 'et-min', 'et-max', 'et-std']
            values = [et_mean, et_min, et_max, et_std]
            for m, v in zip(metrics, values):
                data = {
                    **base_data, 
                    'metric': m,
                    'value': v,
                }
                df = append_row(df, data)
            
            # plot_tiff_arc_shroud(shroud, signals=info['MMPhase0'], frames=frames)

# Save results.
dataset = 'VALKIM-PP-STATIC'
tset = ds.load(dataset, 'training')
evalpath = os.path.join(tset.path, 'data', 'evaluation')
os.makedirs(evalpath, exist_ok=True)
save_csv(df.astype(cols), os.path.join(evalpath, 'breathing-frames.csv'))
print(f"Saved to {evalpath}")
