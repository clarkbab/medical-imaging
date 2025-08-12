import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi.geometry import foreground_fov_width, largest_cc
from mymi.metrics import mean_intensity, snr
from mymi.regions import regions_to_list
from mymi.typing import *
from mymi.utils import *

def create_region_summary(
    dataset: str,
    region_ids: RegionIDs = 'all') -> None:
    logging.arg_log('Creating region summaries', ('dataset', 'region_ids'), (dataset, region_ids))
    set = NiftiDataset(dataset)
    region_ids = regions_to_list(region_ids, literals={ 'all': set.list_regions })

    for region in tqdm(region_ids):
        # Check if there are patients with this region.
        n_pats = len(set.list_patients(region_ids=region_ids))
        if n_pats == 0:
            # Allows us to pass all regions from Spartan 'array' job.
            logging.error(f"No patients with region '{region}' found for dataset '{set}'.")
            return

        # Generate counts report.
        df = get_region_summary(dataset, region, labels='all')

        # Save report.
        filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

def get_region_summary(
    dataset: str,
    region: RegionID,
    labels: Literal['included', 'excluded', 'all'] = 'included') -> pd.DataFrame:
    # List patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(region_ids=region)

    cols = {
        'dataset': str,
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
            'dataset': dataset,
            'patient-id': pat_id,
        }

        # Add OAR volume.
        data['metric'] = 'volume-mm3'
        data['value'] = label.sum() * np.prod(pat.ct_spacing)
        df = append_row(df, data)

        # Add OAR width.
        fov_w = foreground_fov_width(label, offset=pat.ct_offset, spacing=pat.ct_spacing, use_patient_coords=True)
        if fov_w is not None:
            for i, a in enumerate(axes):
                data['metric'] = f'{region}-fov-width-mm-{a}'
                data['value'] = fov_w[i]
                df = append_row(df, data)

        # Add 'connected' metrics.
        data['metric'] = 'connected'
        lcc_label = largest_cc(label)
        data['value'] = 1 if lcc_label.sum() == label.sum() else 0
        df = append_row(df, data)
        data['metric'] = 'connected-largest-p'
        data['value'] = lcc_label.sum() / label.sum()
        df = append_row(df, data)

        # Add fov of largest connected component.
        fov_width_lcc = fov(lcc_label, offset=pat.ct_offset, spacing=pat.ct_spacing, use_patient_coords=True)
        if fov_width_lcc is not None:
            for i, a in enumerate(axes):
                data['metric'] = f'connected-{region}-fov-mm-{a}'
                data['value'] = fov_width_lcc[i]
                df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def load_region_summary(
    dataset: Union[str, List[str]],
    labels: Literal['included', 'excluded', 'all'] = 'included',
    region_ids: RegionIDs = 'all',
    pivot: bool = False,
    raise_error: bool = True) -> Optional[pd.DataFrame]:
    datasets = arg_to_list(dataset, str)
    region_ids = regions_to_list(region_ids)

    # Load summary.
    dfs = []
    for dataset in datasets:
        # Get regions if 'None'.
        set = NiftiDataset(dataset)
        exc_df = set.excluded_labels
        if regions is None:
            dataset_regions = set.list_regions()
        else:
            dataset_regions = regions

        for region in dataset_regions:
            filepath = os.path.join(set.path, 'reports', 'region-summaries', f'{region}.csv')
            if not os.path.exists(filepath):
                if raise_error:
                    raise ValueError(f"Summary not found for region '{region}', dataset '{set}'.")
                else:
                    # Skip this region.
                    continue

            # Add CSV.
            df = pd.read_csv(filepath, dtype={ 'patient-id': str })
            df.insert(1, 'region', region)

            # Filter by 'excluded-labels.csv'.
            rexc_df = None if exc_df is None else exc_df[exc_df['region'] == region] 
            if labels != 'all':
                if rexc_df is None:
                    raise ValueError(f"No 'excluded-labels.csv' specified for '{set}', should pass labels='all'.")
            if labels == 'included':
                df = df.merge(rexc_df, on=['patient-id', 'region'], how='left', indicator=True)
                df = df[df._merge == 'left_only'].drop(columns='_merge')
            elif labels == 'excluded':
                df = df.merge(rexc_df, on=['patient-id', 'region'], how='left', indicator=True)
                df = df[df._merge == 'both'].drop(columns='_merge')

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