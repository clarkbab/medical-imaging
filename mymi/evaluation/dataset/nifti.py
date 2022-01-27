import hashlib
import json
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

from mymi import config
from mymi import dataset as ds
from mymi.geometry import get_box, get_extent_centre
from mymi.loaders import Loader
from mymi.metrics import dice, distances, extent_centre_distance, extent_distance
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction.dataset.nifti import load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi.regions import get_patch_size
from mymi import types

def get_patient_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    truncate_spine: bool = False) -> Dict[str, float]:
    # Get pred/ground truth.
    if region != 'SpinalCord':
        truncate_spine = False
    pred = load_patient_localiser_prediction(dataset, pat_id, localiser, truncate_spine=truncate_spine)
    set = ds.get(dataset, 'nifti')
    label = set.patient(pat_id).region_data(regions=region)[region].astype(np.bool)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing()
    if pred.sum() == 0:
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
    if pred.sum() == 0:
        ec_dist = (np.nan, np.nan, np.nan)
    else:
        ec_dist = extent_centre_distance(pred, label, spacing)

    data['extent-centre-dist-x'] = ec_dist[0]
    data['extent-centre-dist-y'] = ec_dist[1]
    data['extent-centre-dist-z'] = ec_dist[2]

    # Second stage patch distance.
    if pred.sum() == 0:
        e_dist = (np.nan, np.nan, np.nan)
    else:
        # Create second stage patch.
        centre = get_extent_centre(pred)
        size = get_patch_size(region, spacing)
        min, max = get_box(centre, size)

        # Squash to label size.
        min = np.clip(min, a_min=0, a_max=None)
        for i in range(len(max)):
            if max[i] > label.shape[i] - 1:
                max[i] = label.shape[i] - 1

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
    
def create_patient_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    df: Optional[pd.DataFrame] = None,
    truncate_spine: bool = False) -> Optional[pd.DataFrame]:

    # Define dataframe columns.
    cols = {
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }

    # Create/update dataframe if not provided.
    if df is None:
        set = ds.get(dataset, 'nifti')
        filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
        if os.path.exists(filepath):
            # Load dataframe.
            eval_df = load_localiser_evaluation(dataset, localiser)
        else:
            # Create dataframe.
            eval_df = pd.DataFrame(columns=cols.keys())
    else:
        eval_df = df

    # Get metrics.
    if region != 'SpinalCord':
        truncate_spine = False
    metrics = get_patient_localiser_evaluation(dataset, pat_id, region, localiser, truncate_spine=truncate_spine)

    # Add/update each metric.
    for metric, value in metrics.items():
        exists = len(eval_df[(eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric)]) != 0
        if not exists:
            # Add metric.
            data = {
                'patient-id': pat_id, 
                'region': region,
                'metric': metric,
                'value': value
            }
            eval_df = eval_df.append(data, ignore_index=True)
        else:
            # Update metric.
            eval_df.loc[(eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric), 'value'] = value

    if df is None:
        # Set column types.
        eval_df = eval_df.astype(cols)

        # Save evaluation.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        eval_df.to_csv(filepath, index=False)
    else:
        return eval_df

def create_localiser_evaluation(
    dataset: str,
    region: str,
    localiser: types.ModelName,
    truncate_spine: bool = False) -> None:
    # Load localiser.
    localiser = Localiser.load(*localiser)
    logging.info(f"Evaluating localiser predictions for NIFTI dataset '{dataset}', region '{region}', localiser '{localiser.name}'.")

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
        if region != 'SpinalCord':
            truncate_spine = False
        df = create_patient_localiser_evaluation(dataset, pat, region, localiser, df=df, truncate_spine=truncate_spine)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_localiser_evaluation_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    truncate_spine: bool = False) -> None:
    # Get unique name.
    localiser = Localiser.replace_best(*localiser)

    logging.info(f"Evaluating localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}'.")

    # Create test loader.
    sets = [ds.get(d, 'training') for d in datasets]
    _, _, test_loader = Loader.build_loaders(sets, region, num_folds=num_folds, test_fold=test_fold)

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for dataset_b, pat_id_b in tqdm(iter(test_loader)):
        if region != 'SpinalCord':
            truncate_spine = False
        if type(pat_id_b) == torch.Tensor:
            pat_id_b = pat_id_b.tolist()
        for dataset, pat_id in zip(dataset_b, pat_id_b):
            df = create_patient_localiser_evaluation(dataset, pat_id, region, localiser, df=df, truncate_spine=truncate_spine)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    folder = hashlib.sha1(json.dumps(datasets).encode('utf-8')).hexdigest()
    filename = f'eval-folds-{num_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.files, 'evaluation', 'localiser', *localiser, folder, f'{filename}.csv')
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

def load_localiser_evaluation_from_loader(
    datasets: Union[str, List[str]],
    localiser: types.ModelName) -> np.ndarray:
    set = ds.get(dataset, 'nifti')
    localiser = Localiser.replace_best(*localiser)
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Evaluation for dataset '{set}', localiser '{localiser}' not found.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def get_patient_segmenter_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)
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

    return data
    
def create_patient_segmenter_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:

    # Define dataframe columns.
    cols = {
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }

    # Create/update dataframe if not provided.
    if df is None:
        set = ds.get(dataset, 'nifti')
        filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, *segmenter, 'eval.csv') 
        if os.path.exists(filepath):
            # Load dataframe.
            eval_df = load_segmenter_evaluation(dataset, localiser, segmenter)
        else:
            # Create dataframe.
            eval_df = pd.DataFrame(columns=cols.keys())
    else:
        eval_df = df

    # Get metrics.
    metrics = get_patient_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter)

    # Add/update each metric.
    for metric, value in metrics.items():
        exists = len(eval_df[(eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric)]) != 0
        if not exists:
            # Add metric.
            data = {
                'patient-id': pat_id, 
                'region': region,
                'metric': metric,
                'value': value
            }
            eval_df = eval_df.append(data, ignore_index=True)
        else:
            # Update metric.
            eval_df.loc[(eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric), 'value'] = value

    if df is None:
        # Set column types.
        eval_df = eval_df.astype(cols)

        # Save evaluation.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        eval_df.to_csv(filepath, index=False)
    else:
        return eval_df

def create_segmenter_evaluation(
    dataset: str,
    region: str,
    localiser: types.Model,
    segmenter: types.Model) -> None:
    localiser = Localiser.replace_best(*localiser)
    segmenter = Segmenter.replace_best(*segmenter)
    logging.info(f"Evaluating segmenter predictions for NIFTI dataset '{dataset}', region '{region}', localiser '{localiser}' and segmenter '{segmenter}'.")

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
        # Get metrics.
        metrics = get_patient_segmenter_evaluation(dataset, pat, region, localiser, segmenter)

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
    filepath = os.path.join(set.path, 'evaluation', 'segmenter', *localiser, *segmenter, 'eval.csv') 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_segmenter_evaluation_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    # Get unique names.
    localiser = Localiser.replace_best(*localiser)
    segmenter = Segmenter.replace_best(*segmenter)

    logging.info(f"Evaluating segmenter predictions for NIFTI dataset '{datasets}', region '{region}', localiser '{localiser}' and segmenter '{segmenter}'.")

    # Create test loader.
    sets = [ds.get(d, 'training') for d in datasets]
    _, _, test_loader = Loader.build_loaders(sets, region, num_folds=num_folds, test_fold=test_fold)

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for dataset_b, pat_id_b in tqdm(iter(test_loader)):
        if type(pat_id_b) == torch.Tensor:
            pat_id_b = pat_id_b.tolist()
        for dataset, pat_id in zip(dataset_b, pat_id_b):
            df = create_patient_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter, df=df)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    folder = hashlib.sha1(json.dumps(datasets).encode('utf-8')).hexdigest()
    filename = f'eval-folds-{num_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.files, 'evaluation', 'segmenter', *localiser, *segmenter, folder, f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_segmenter_evaluation(
    dataset: str,
    localiser: Tuple[str, str, str],
    segmenter: Tuple[str, str, str]) -> np.ndarray:
    localiser = Localiser.replace_best(*localiser)
    segmenter = Segmenter.replace_best(*segmenter)
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'evaluation', 'segmenter', *localiser, *segmenter, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Segmenter evaluation for dataset '{set}', localiser '{localiser}' and segmenter '{segmenter}' not found.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data
