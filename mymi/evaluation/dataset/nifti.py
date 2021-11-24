import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from mymi import cache
from mymi import dataset as ds
from mymi.geometry import get_box, get_extent_centre
from mymi.metrics import dice, distances, extent_centre_distance, extent_distance
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction.dataset.nifti import load_localiser_prediction
from mymi.regions import get_patch_size
from mymi import types

def get_patient_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: Tuple[str, str, str]) -> Dict[str, float]:
    # Get pred/ground truth.
    localiser = Localiser.replace_best(*localiser)
    pred = load_localiser_prediction(dataset, pat_id, localiser)
    set = ds.get(dataset, 'nifti')
    label = set.patient(pat_id).region_data(regions=region)[region].astype(np.bool)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing()
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

    data['assd'] = dists['assd']
    data['surface-hd'] = dists['surface-hd']
    data['surface-95hd'] = dists['surface-95hd']
    data['voxel-hd'] = dists['voxel-hd']
    data['voxel-95hd'] = dists['voxel-95hd']

    # Extent distance.
    if pred.sum() == 0 or label.sum() == 0:
        ec_dist = (np.nan, np.nan, np.nan)
    else:
        ec_dist = extent_centre_distance(pred, label, spacing)

    data['extent-centre-dist-x'] = ec_dist[0]
    data['extent-centre-dist-y'] = ec_dist[1]
    data['extent-centre-dist-z'] = ec_dist[2]

    # Second stage patch distance.
    if pred.sum() == 0 or label.sum() == 0:
        e_dist = (np.nan, np.nan, np.nan)
    else:
        # Create second stage patch.
        centre = get_extent_centre(pred)
        size = get_patch_size(region)
        min, max = get_box(centre, size)

        # Create label from patch.
        patch_label = np.zeros_like(label)
        slices = tuple([slice(l, h + 1) for l, h in zip(min, max)])
        patch_label[slices] = 1

        # Get extent distance.
        e_dist = extent_distance(patch_label, label, spacing)

    data['extent-dist-x'] = e_dist[0]
    data['extent-dist-y'] = e_dist[1]
    data['extent-dist-z'] = e_dist[2]

    return data

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
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Get pred/ground truth.
        pred = load_localiser_prediction(dataset, pat, localiser)
        label = set.patient(pat).region_data(regions=region)[region].astype(np.bool)

        # Get metrics.
        metrics = get_patient_localiser_evaluation(dataset, pat, region, localiser)

        # Add metrics.
        for metric, value in metrics.items():
            data = {
                'patient-id': pat, 
                'region': region,
                'metric': metric,
                'value': value
            }
            df = df.append(data, ignore_index=True)

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
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

