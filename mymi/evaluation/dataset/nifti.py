import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple

from mymi import cache
from mymi import dataset as ds
from mymi.metrics import dice, distances
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction.dataset.nifti import get_localiser_prediction
from mymi import types

def evaluate_localiser_predictions(
    dataset: str,
    localiser: Tuple[str, str, str],
    region: str) -> None:
    # Load dataset.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=region)

    # Create dataframe.
    cols = {
        'patient-id': str,
        'metric': str,
        region: float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Get pred/ground truth.
        pred = get_localiser_prediction(dataset, pat, localiser)
        label = set.patient(pat).region_data(regions=region)[region].astype(np.bool)

        # Add metrics.
        metrics = [
            'dice',
            'assd',
            'surface-hd',
            'surface-95hd',
            'voxel-hd',
            'voxel-95hd'
        ]
        data = {}
        for metric in metrics:
            data[metric] = {
                'patient-id': pat,
                'metric': metric
            }

        # Dice.
        dsc = dice(pred, label)
        data['dice'][region] = dsc
        df = df.append(data['dice'], ignore_index=True)

        # Distances.
        spacing = set.patient(pat).ct_spacing()
        try:
            dists = distances(pred, label, spacing)
        except ValueError:
            dists = {
                'assd': np.nan,
                'surface-hd': np.nan,
                'surface-95hd': np.nan,
                'voxel-hd': np.nan,
                'voxel-95hd': np.nan
            }

        data['assd'][region] = dists['assd']
        data['surface-hd'][region] = dists['surface-hd']
        data['surface-95hd'][region] = dists['surface-95hd']
        data['voxel-hd'][region] = dists['voxel-hd']
        data['voxel-95hd'][region] = dists['voxel-95hd']
        df = df.append(data['assd'], ignore_index=True)
        df = df.append(data['surface-hd'], ignore_index=True)
        df = df.append(data['surface-95hd'], ignore_index=True)
        df = df.append(data['voxel-hd'], ignore_index=True)
        df = df.append(data['voxel-95hd'], ignore_index=True)

    # Set column types.
    df = df.astype(cols)

    # Set index.
    df = df.set_index('patient-id')

    # Save evaluation.
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def get_localiser_evaluation(
    dataset: str,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Evaluation for dataset '{set}', localiser '{localiser}' not found.")
    data = pd.read_csv(filepath)
    return data
