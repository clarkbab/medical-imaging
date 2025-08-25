import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from mymi.datasets import NiftiDataset
from mymi.utils import *

def create_ct_summary(dataset: str) -> None:
    # Get regions.
    set = NiftiDataset(dataset)

    # Get summary.
    df = get_ct_summary(dataset)

    # Save summary.
    filepath = os.path.join(set.path, 'reports', 'ct-summary.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def get_ct_summary(dataset: str) -> pd.DataFrame:
    # Get patients.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()

    cols = {
        'dataset': str,
        'patient-id': str,
        'axis': int,
        'size': int,
        'spacing': float,
        'fov': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat_id in tqdm(pat_ids):
        # Load values.
        patient = set.patient(pat_id)
        size = patient.ct_size
        spacing = patient.ct_spacing

        # Calculate FOV.
        fov = np.array(size) * spacing

        for axis in range(len(size)):
            data = {
                'dataset': dataset,
                'patient-id': pat_id,
                'axis': axis,
                'size': size[axis],
                'spacing': spacing[axis],
                'fov': fov[axis]
            }
            df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def load_ct_summary(datasets: Union[str, List[str]]) -> pd.DataFrame:
    datasets = arg_to_list(datasets, str)

    dfs = []
    for dataset in datasets:
        # Get regions.
        set = NiftiDataset(dataset)

        filepath = os.path.join(set.path, 'reports', 'ct-summary.csv')
        if not os.path.exists(filepath):
            raise ValueError(f"CT summary report doesn't exist for dataset '{dataset}'.")
        df = pd.read_csv(filepath)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)
    df = df.astype({ 'patient-id': str })

    return df
