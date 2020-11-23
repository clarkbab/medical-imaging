import argparse
import os
import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

DATA_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1')
DATA_SOURCE = os.path.join(DATA_ROOT, 'raw')
DATA_SUMMARY = os.path.join(DATA_ROOT, 'summary')

# Parse args.
parser = argparse.ArgumentParser(description='Get dataset summary')
parser.add_argument('-o', '--overwrite', required=False, default=False, action='store_true', help='overwrite existing summary data')
parser.add_argument('-t', '--test', required=False, default=False, action='store_true', help='use 5 samples only' )
args = parser.parse_args()

if args.overwrite:
    # Create info table.
    info_cols = {
        'dim-x': np.uint16,
        'dim-y': np.uint16,
        'dim-z': np.uint16,
        'fov-x': 'float64',
        'fov-y': 'float64',
        'fov-z': 'float64',
        'hu-min': 'float64',
        'hu-max': 'float64',
        'offset-x': 'float64',
        'offset-y': 'float64',
        'pat-id': 'object',
        'res-x': 'float64',
        'res-y': 'float64',
        'res-z': 'float64',
        'roi-num': np.uint16,
        'scale-int': 'float64',
        'scale-slope': 'float64'
    }
    info_df = pd.DataFrame(columns=np.sort(list(info_cols.keys())))
    info_df = info_df.astype(info_cols)

    # Create label table.
    label_info_cols = {
        'roi-label': 'object',
        'count': np.uint16
    }
    label_info_df = pd.DataFrame(columns=np.sort(list(label_info_cols.keys())))
    label_info_df = label_info_df.astype(label_info_cols)

    # Load patients.
    pat_dirs = np.sort(os.listdir(DATA_SOURCE))
    pat_paths = [os.path.join(DATA_SOURCE, d) for d in pat_dirs]
    if args.test:
        pat_paths = pat_paths[:5]

    for pat_path in tqdm(pat_paths):
        # Load dicoms and extract info.
        ct_dicoms = utils.load_ct_dicoms(pat_path)
        ct_info = utils.get_ct_info(ct_dicoms)
        rtstruct_dicom = utils.load_rtstruct_dicom(pat_path)
        rtstruct_info = utils.get_rtstruct_info(rtstruct_dicom)

        # Get patient ID from path.
        pat_id = pat_path.split(os.sep)[-1]

        # Summarise CT data.
        assert len(ct_info['dim-x'].unique()) == 1
        assert len(ct_info['dim-y'].unique()) == 1
        assert len(ct_info['offset-x'].unique()) == 1
        assert len(ct_info['offset-y'].unique()) == 1
        assert len(ct_info['res-x'].unique()) == 1
        assert len(ct_info['res-y'].unique()) == 1
        assert len(ct_info['scale-int'].unique()) == 1
        assert len(ct_info['scale-slope'].unique()) == 1
        res_zs = np.sort(np.diff(ct_info['offset-z']))
        if len(np.unique(res_zs)) != 1:
            print(f"Inconsistent z-spacing, choosing smallest resolution '{res_zs[0]}'")

        ct_info = {
            'dim-x': ct_info['dim-x'][0],
            'dim-y': ct_info['dim-y'][0],
            'dim-z': len(ct_info),
            'hu-min': ct_info['hu-min'].min(),
            'hu-max': ct_info['hu-max'].max(),
            'offset-x': ct_info['offset-x'][0],
            'offset-y': ct_info['offset-y'][0],
            'res-x': ct_info['res-x'][0],
            'res-y': ct_info['res-y'][0],
            'res-z': res_zs[0], 
            'scale-int': ct_info['scale-int'][0],
            'scale-slope': ct_info['scale-slope'][0],
        } 

        # Summary RTSTRUCT data.
        rtstruct_summary_info = {
            'roi-num': len(rtstruct_info)
        }

        # Add info to summary.
        merged_info = {'pat-id': pat_id, **ct_info, **rtstruct_summary_info}
        info_df = info_df.append(merged_info, ignore_index=True)

        # Add label counts.
        rtstruct_info['count'] = 1
        label_info_df = label_info_df.merge(rtstruct_info, how='outer', on='roi-label')
        label_info_df['count'] = (label_info_df['count_x'].fillna(0) + label_info_df['count_y'].fillna(0)).astype(np.uint16)
        label_info_df = label_info_df.drop(['count_x', 'count_y'], axis=1)
        
    # Add field-of-view columns.
    info_df['fov-x'] = info_df['dim-x'] * info_df['res-x']
    info_df['fov-y'] = info_df['dim-y'] * info_df['res-y']
    info_df['fov-z'] = info_df['dim-z'] * info_df['res-z']

    # Save info.
    os.makedirs(DATA_SUMMARY, exist_ok=True)
    summary_path = os.path.join(DATA_SUMMARY, 'summary.csv')
    info_df.to_csv(summary_path)

    # Create label summary.
    label_summary_path = os.path.join(DATA_SUMMARY, 'label_summary.csv')
    label_info_df.to_csv(label_summary_path)

# Load and print summary.
summary_path = os.path.join(DATA_SUMMARY, 'summary.csv') 
info_df = pd.read_csv(summary_path, index_col=0)
print(info_df.describe())

label_summary_path = os.path.join(DATA_SUMMARY, 'label_summary.csv') 
label_info_df = pd.read_csv(label_summary_path, index_col=0)
print(label_info_df)