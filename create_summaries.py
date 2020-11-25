import argparse
import os
import numpy as np
import pandas as pd
import utils
from tqdm import tqdm

COMP_PRECISION = 2
DATA_ROOT = os.path.join(os.sep, 'media', 'brett', 'data', 'HEAD-NECK-RADIOMICS-HN1')
DATA_SOURCE = os.path.join(DATA_ROOT, 'raw')
SUMMARY_ROOT = os.path.join(DATA_ROOT, 'summaries')
CT_SUMMARY_PATH = os.path.join(SUMMARY_ROOT, 'ct_summary.csv')
RTSTRUCT_SUMMARY_PATH = os.path.join(SUMMARY_ROOT, 'rtstruct_summary.csv')

# Parse args.
parser = argparse.ArgumentParser(description='Get dataset summary')
parser.add_argument('-o', '--overwrite', required=False, default=False, action='store_true', help='overwrite existing summary data')
parser.add_argument('-n', '--numpats', type=int, required=False, help='run on a smaller number of patients' )
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
        'num-empty': np.uint16,
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
    ct_summary_df = pd.DataFrame(columns=np.sort(list(info_cols.keys())))
    ct_summary_df = ct_summary_df.astype(info_cols)

    # Create label table.
    label_info_cols = {
        'roi-label': 'object',
        'count': np.uint16
    }
    rtstruct_summary_df = pd.DataFrame(columns=np.sort(list(label_info_cols.keys())))
    rtstruct_summary_df = rtstruct_summary_df.astype(label_info_cols)

    # Load patients.
    pat_dirs = np.sort(os.listdir(DATA_SOURCE))
    pat_paths = sorted([os.path.join(DATA_SOURCE, d) for d in pat_dirs])
    if args.numpats:
        pat_paths = pat_paths[:args.numpats]

    for pat_path in tqdm(pat_paths):
        # Load dicoms and extract info.
        ct_dicoms = utils.load_ct_dicoms(pat_path)
        ct_dicoms_info = utils.get_ct_info(ct_dicoms)
        rtstruct_dicom = utils.load_rtstruct_dicom(pat_path)
        rtstruct_dicom_info = utils.get_rtstruct_info(rtstruct_dicom)

        # Get patient ID from path.
        pat_id = pat_path.split(os.sep)[-1]

        # Check for consistency among patient scans.
        assert len(ct_dicoms_info['dim-x'].unique()) == 1
        assert len(ct_dicoms_info['dim-y'].unique()) == 1
        assert len(ct_dicoms_info['offset-x'].unique()) == 1
        assert len(ct_dicoms_info['offset-y'].unique()) == 1
        assert len(ct_dicoms_info['res-x'].unique()) == 1
        assert len(ct_dicoms_info['res-y'].unique()) == 1
        assert len(ct_dicoms_info['scale-int'].unique()) == 1
        assert len(ct_dicoms_info['scale-slope'].unique()) == 1

        # Calculate res-z - this will be the smallest available diff.
        res_zs = np.sort([round(i, COMP_PRECISION) for i in np.diff(ct_dicoms_info['offset-z'])])
        res_z = res_zs[0]

        # Calculate fov-z and dim-z.
        fov_z = ct_dicoms_info['offset-z'].max() - ct_dicoms_info['offset-z'].min()
        dim_z = int(fov_z / res_z)

        # Calculate number of empty slices.
        num_slices = len(ct_dicoms_info)
        num_empty = dim_z - len(ct_dicoms_info) + 1

        ct_summary_info = {
            'dim-x': ct_dicoms_info['dim-x'][0],
            'dim-y': ct_dicoms_info['dim-y'][0],
            'dim-z': dim_z,
            'hu-min': ct_dicoms_info['hu-min'].min(),
            'hu-max': ct_dicoms_info['hu-max'].max(),
            'num-empty': num_empty,
            'offset-x': ct_dicoms_info['offset-x'][0],
            'offset-y': ct_dicoms_info['offset-y'][0],
            'res-x': ct_dicoms_info['res-x'][0],
            'res-y': ct_dicoms_info['res-y'][0],
            'res-z': res_z, 
            'scale-int': ct_dicoms_info['scale-int'][0],
            'scale-slope': ct_dicoms_info['scale-slope'][0],
        } 

        # Summary RTSTRUCT data.
        rtstruct_summary_info = {
            'roi-num': len(rtstruct_dicom_info)
        }

        # Add info to summary.
        merged_info = {
            'pat-id': pat_id,
            **ct_summary_info,
            **rtstruct_summary_info
        }
        ct_summary_df = ct_summary_df.append(merged_info, ignore_index=True)

        # Add label counts.
        rtstruct_dicom_info['count'] = 1
        rtstruct_summary_df = rtstruct_summary_df.merge(rtstruct_dicom_info, how='outer', on='roi-label')
        rtstruct_summary_df['count'] = (rtstruct_summary_df['count_x'].fillna(0) + rtstruct_summary_df['count_y'].fillna(0)).astype(np.uint16)
        rtstruct_summary_df = rtstruct_summary_df.drop(['count_x', 'count_y'], axis=1)
        
    # Add field-of-view columns.
    ct_summary_df['fov-x'] = ct_summary_df['dim-x'] * ct_summary_df['res-x']
    ct_summary_df['fov-y'] = ct_summary_df['dim-y'] * ct_summary_df['res-y']
    ct_summary_df['fov-z'] = ct_summary_df['dim-z'] * ct_summary_df['res-z']

    # Set index.
    ct_summary_df = ct_summary_df.set_index('pat-id')

    # Save info.
    os.makedirs(SUMMARY_ROOT, exist_ok=True)
    ct_summary_df.to_csv(CT_SUMMARY_PATH)

    # Save label summary.
    rtstruct_summary_df = rtstruct_summary_df.sort_values('count', ascending=False).reset_index().drop('index', axis=1)
    rtstruct_summary_df.to_csv(RTSTRUCT_SUMMARY_PATH)

# Load and print summary.
ct_summary_df = pd.read_csv(CT_SUMMARY_PATH, index_col=0)
print(ct_summary_df.describe())

rtstruct_summary_df = pd.read_csv(RTSTRUCT_SUMMARY_PATH, index_col=0)
print(rtstruct_summary_df)