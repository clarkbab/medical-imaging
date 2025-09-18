import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi.geometry import foreground_fov_width, foreground_fov, largest_cc
from mymi.metrics import mean_intensity, snr
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

def create_region_counts(dataset: DatasetID) -> None:
    pr_df = get_region_counts(dataset)
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    save_csv(pr_df, filepath)

def create_region_summary(
    dataset: str,
    region_ids: RegionIDs = 'all') -> None:
    logging.arg_log('Creating region summaries', ('dataset', 'region_ids'), (dataset, region_ids))
    set = NiftiDataset(dataset)
    region_ids = regions_to_list(region_ids, literals={ 'all': set.list_regions })

    for region in tqdm(region_ids):
        filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
        assert_writeable(filepath)

        # Check if there are patients with this region.
        n_pats = len(set.list_patients(region_ids=region_ids))
        if n_pats == 0:
            # Allows us to pass all regions from Spartan 'array' job.
            logging.error(f"No patients with region '{region}' found for dataset '{set}'.")
            return

        # Generate counts report.
        df = get_region_summary(dataset, region, labels='all')

        # Save report.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

def get_region_counts(dataset: str) -> pd.DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Add rows.
    for pat_id in tqdm(pat_ids):
        pat_regions = set.patient(pat_id).list_regions()
        for pat_region in pat_regions:
            data = {
                'patient-id': pat_id,
                'region': pat_region
            }
            df = append_row(df, data)

    return df

def get_region_summary(
    dataset: str,
    region: RegionID,
    labels: Literal['included', 'excluded', 'all'] = 'included') -> pd.DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(region_ids=region)

    cols = {
        'patient-id': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    axes = ['x', 'y', 'z']
    extrema = ['min', 'max']
    for pat_id in tqdm(pat_ids):
        pat = set.patient(pat_id)
        label = pat.region_data(region_ids=region)[region]

        data = {
            'patient-id': pat_id,
        }

        # Add OAR position.
        fov_l = foreground_fov(label, spacing=pat.ct_spacing, origin=pat.ct_origin, use_patient_coords=True)
        if fov_l is not None:
            for i, a in enumerate(axes):
                for j, e in enumerate(extrema):
                    data['metric'] = f'fov-{e}-mm-{a}'
                    data['value'] = fov_l[j][i]
                    df = append_row(df, data)

        # Add OAR volume.
        data['metric'] = 'volume-mm3'
        data['value'] = label.sum() * np.prod(pat.ct_spacing)
        df = append_row(df, data)

        # Add OAR width.
        fov_w = foreground_fov_width(label, origin=pat.ct_origin, spacing=pat.ct_spacing, use_patient_coords=True)
        if fov_w is not None:
            for i, a in enumerate(axes):
                data['metric'] = f'fov-width-mm-{a}'
                data['value'] = fov_w[i]
                df = append_row(df, data)

        # Add boolean for connectedness.
        data['metric'] = 'connected'
        lcc_label = largest_cc(label)
        data['value'] = 1 if lcc_label.sum() == label.sum() else 0
        df = append_row(df, data)

        # Add volume of largest connected component as proportion of total foreground volume.
        data['metric'] = 'connected-largest-p'
        data['value'] = lcc_label.sum() / label.sum()
        df = append_row(df, data)

        # Add position of largest connected component.
        fov_l = foreground_fov(lcc_label, spacing=pat.ct_spacing, origin=pat.ct_origin, use_patient_coords=True)
        if fov_l is not None:
            for i, a in enumerate(axes):
                for j, e in enumerate(extrema):
                    data['metric'] = f'fov-{e}-mm-{a}'
                    data['value'] = fov_l[j][i]
                    df = append_row(df, data)

        # Add fov of largest connected component.
        fov_width_lcc = foreground_fov_width(lcc_label, origin=pat.ct_origin, spacing=pat.ct_spacing, use_patient_coords=True)
        if fov_width_lcc is not None:
            for i, a in enumerate(axes):
                data['metric'] = f'connected-fov-mm-{a}'
                data['value'] = fov_width_lcc[i]
                df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def load_region_counts(
    dataset: DatasetID,
    region_ids: RegionIDs = 'all',
    exists_only: bool = False) -> Union[pd.DataFrame, bool]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
        else:
            df = load_csv(filepath)
            if region_ids != 'all':
                region_ids = regions_to_list(region_ids)
                df = df[df['region'].isin(region_ids)]
            return df
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Patient regions report doesn't exist for dataset '{dataset}'.")

def load_region_summary(
    dataset: Union[str, List[str]],
    labels: Literal['included', 'excluded', 'all'] = 'included',
    region_ids: RegionIDs = 'all',
    pivot: bool = False,
    raise_error: bool = True) -> Optional[pd.DataFrame]:
    datasets = arg_to_list(dataset, str)

    # Load summary.
    dfs = []
    for dataset in datasets:
        set = NiftiDataset(dataset)
        ds_region_ids = regions_to_list(region_ids, literals={ 'all': set.list_regions })

        for region in ds_region_ids:
            filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
            if not os.path.exists(filepath):
                if raise_error:
                    raise ValueError(f"Summary not found for region '{region}', dataset '{set}'.")
                else:
                    # Skip this region.
                    continue

            # Load CSV.
            df = pd.read_csv(filepath, dtype={ 'patient-id': str })
            df.insert(0, 'dataset', dataset)
            df.insert(2, 'region', region)

            # Add region summary.
            dfs.append(df)

    # Concatenate loaded files.
    if len(dfs) == 0:
        return None
    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)

    # If pivot.
    if pivot:
        df = df.pivot(index=['dataset', 'patient-id', 'region'], columns='metric', values='value')
        df = df.reset_index()

    return df