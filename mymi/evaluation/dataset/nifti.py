import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple

from mymi import cache
from mymi import dataset as ds
from mymi.metrics import dice, distances, extent_centre_distance
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction.dataset.nifti import load_localiser_prediction
from mymi import types

def create_localiser_evaluation(
    dataset: str,
    region: str,
    localiser: Tuple[str, str, str]) -> None:
    localiser = Localiser.replace_best(*localiser)
    logging.info(f"Evaluating localiser predictions for NIFTI dataset '{dataset}', region '{region}', localiser '{localiser}'.")

    # Load dataset.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=region)

    # Create dataframe.
    cols = {
        'patient-id': str,
        'metric': str,
        'region': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Get pred/ground truth.
        pred = load_localiser_prediction(dataset, pat, localiser)
        label = set.patient(pat).region_data(regions=region)[region].astype(np.bool)

        # Add metrics.
        metrics = [
            'dice',
            'assd',
            'extent-centre-x',
            'extent-centre-y',
            'extent-centre-z',
            'surface-hd',
            'surface-95hd',
            'voxel-hd',
            'voxel-95hd'
        ]
        data = {}
        for metric in metrics:
            data[metric] = {
                'patient-id': pat,
                'metric': metric,
                'region': region
            }

        # Dice.
        dsc = dice(pred, label)
        data['dice']['value'] = dsc
        df = df.append(data['dice'], ignore_index=True)

        # Distances.
        spacing = set.patient(pat).ct_spacing()
        if pred.sum() == 0 or label.sum() == 0:
            dists = {
                'assd': np.nan,
                'surface-hd': np.nan,
                'surface-95hd': np.nan,
                'voxel-hd': np.nan,
                'voxel-95hd': np.nan
            }
        else:
            dists = distances(pred, label, spacing)

        data['assd']['value'] = dists['assd']
        data['surface-hd']['value'] = dists['surface-hd']
        data['surface-95hd']['value'] = dists['surface-95hd']
        data['voxel-hd']['value'] = dists['voxel-hd']
        data['voxel-95hd']['value'] = dists['voxel-95hd']
        df = df.append(data['assd'], ignore_index=True)
        df = df.append(data['surface-hd'], ignore_index=True)
        df = df.append(data['surface-95hd'], ignore_index=True)
        df = df.append(data['voxel-hd'], ignore_index=True)
        df = df.append(data['voxel-95hd'], ignore_index=True)

        # Extent distance.
        if pred.sum() == 0 or label.sum() == 0:
            dist = (np.nan, np.nan, np.nan)
        else:
            dist = extent_centre_distance(pred, label, spacing)

        data['extent-centre-x']['value'] = dist[0]
        data['extent-centre-y']['value'] = dist[1]
        data['extent-centre-z']['value'] = dist[2]
        df = df.append(data['extent-centre-x'], ignore_index=True)
        df = df.append(data['extent-centre-y'], ignore_index=True)
        df = df.append(data['extent-centre-z'], ignore_index=True)

        # Extent cover.
        data['extent-cover'] = cover

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_localiser_evaluation(
    dataset: str,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    set = ds.get(dataset, 'nifti')
    localiser = Localiser.replace_best(*localiser)
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Evaluation for dataset '{set}', localiser '{localiser}' not found.")
    data = pd.read_csv(filepath)
    return data
