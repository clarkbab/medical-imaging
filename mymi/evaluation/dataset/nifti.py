from mymi.transforms.crop_or_pad import crop_foreground_3D
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Literal, Optional, Tuple, Union

from mymi import config
from mymi import dataset as ds
from mymi.geometry import get_box, get_encaps_dist_mm, get_extent_centre
from mymi.loaders import Loader
from mymi.metrics import dice, distances, extent_centre_distance
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction.dataset.nifti import load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi.regions import get_patch_size
from mymi import types
from mymi.utils import encode

def get_patient_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_patient_localiser_prediction(dataset, pat_id, localiser)
    set = ds.get(dataset, 'nifti')
    label = set.patient(pat_id).region_data(regions=region)[region].astype(np.bool)

    # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
    if region == 'SpinalCord':
        z_min_pred = np.nonzero(pred)[2].min()
        z_min_label = np.nonzero(label)[2].min()
        z_min = np.max([z_min_label, z_min_pred])

        # Crop pred/label foreground voxels.
        crop = ((0, 0, z_min), label.shape)
        pred = crop_foreground_3D(pred, crop)
        label = crop_foreground_3D(label, crop)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing
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
        # Get second stage patch coordinates.
        centre = get_extent_centre(pred)
        size = get_patch_size(region, spacing)
        min, max = get_box(centre, size)

        # Clip patch coordinates to label size.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.clip(max, a_min=None, a_max=label.shape)

        # Convert patch coordinates into label.
        patch_label = np.zeros_like(label)
        slices = tuple([slice(l, h + 1) for l, h in zip(min, max)])
        patch_label[slices] = 1

        # Get extent distance.
        e_dist = get_encaps_dist_mm(patch_label, label, spacing)

    data['encaps-dist-mm-x'] = e_dist[0]
    data['encaps-dist-mm-y'] = e_dist[1]
    data['encaps-dist-mm-z'] = e_dist[2]

    return data
    
def create_patient_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:

    # Define dataframe columns.
    cols = {
        'dataset': str,
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
    metrics = get_patient_localiser_evaluation(dataset, pat_id, region, localiser)

    # Add/update each metric.
    for metric, value in metrics.items():
        exists_df = eval_df[(eval_df['dataset'] == dataset) & (eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric)]
        if len(exists_df) == 0:
            # Add metric.
            data = {
                'dataset': dataset,
                'patient-id': pat_id, 
                'region': region,
                'metric': metric,
                'value': value
            }
            eval_df = eval_df.append(data, ignore_index=True)
        else:
            # Update metric.
            eval_df.loc[(eval_df['dataset'] == dataset) & (eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric), 'value'] = value

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
    localiser: types.ModelName) -> None:
    # Load localiser.
    localiser = Localiser.load(*localiser)
    logging.info(f"Evaluating localiser predictions for NIFTI dataset '{dataset}', region '{region}', localiser '{localiser.name}'.")

    # Load dataset.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=region)

    # Create dataframe.
    cols = {
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        df = create_patient_localiser_evaluation(dataset, pat, region, localiser, df=df)

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
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    # Get unique name.
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    logging.info(f"Evaluating localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', with {num_folds}-fold CV using test folds '{test_folds}'.")

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(num_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    for test_fold in tqdm(test_folds):
        # Create dataframe.
        cols = {
            'fold': int,
            'patient-id': str,
            'region': str,
            'metric': str,
            'value': float
        }
        df = pd.DataFrame(columns=cols.keys())

        # Build test loader.
        _, _, test_loader = Loader.build_loaders(datasets, region, num_folds=num_folds, test_fold=test_fold)

        # Add evaluations to dataframe.
        for dataset_b, pat_id_b in tqdm(iter(test_loader), leave=False):
            if type(pat_id_b) == torch.Tensor:
                pat_id_b = pat_id_b.tolist()
            for dataset, pat_id in zip(dataset_b, pat_id_b):
                df = create_patient_localiser_evaluation(dataset, pat_id, region, localiser, df=df)

        # Add fold.
        df['fold'] = test_fold

        # Set column types.
        df = df.astype(cols)

        # Save evaluation.
        filename = f'eval-folds-{num_folds}-test-{test_fold}'
        filepath = os.path.join(config.directories.evaluation, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

def load_localiser_evaluation(
    dataset: str,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    set = ds.get(dataset, 'nifti')
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    filepath = os.path.join(set.path, 'evaluation', 'localiser', *localiser, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Evaluation for dataset '{set}', localiser '{localiser}' not found.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def load_localiser_evaluation_from_loader(
    datasets: Union[str, List[str]],
    localiser: types.ModelName,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> np.ndarray:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    filename = f'eval-folds-{num_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluation, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
    if not os.path.exists(filepath):
        print(filepath)
        raise ValueError(f"Localiser evaluation for dataset '{datasets}', localiser '{localiser}', {num_folds}-fold CV with test fold {test_fold} not found.")
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

    # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
    if region == 'SpinalCord':
        z_min_pred = np.nonzero(pred)[2].min()
        z_min_label = np.nonzero(label)[2].min()
        z_min = np.max([z_min_label, z_min_pred])

        # Crop pred/label foreground voxels.
        crop = ((0, 0, z_min), label.shape)
        pred = crop_foreground_3D(pred, crop)
        label = crop_foreground_3D(label, crop)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing
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
        'dataset': str,
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
        exists_df = eval_df[(eval_df['dataset'] == dataset) & (eval_df['patient-id'] == pat_id) & (eval_df.region == region) & (eval_df.metric == metric)]
        if len(exists_df) == 0:
            # Add metric.
            data = {
                'dataset': dataset,
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
    localiser: types.ModelName,
    segmenter: types.ModelName) -> None:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)
    logging.info(f"Evaluating segmenter predictions for NIFTI dataset '{dataset}', region '{region}', localiser '{localiser}' and segmenter '{segmenter}'.")

    # Load dataset.
    set = ds.get(dataset, 'nifti')
    pats = set.list_patients(regions=region)

    # Create dataframe.
    cols = {
        'dataset': str,
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
                'dataset': dataset,
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
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    # Get unique name.
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)
    logging.info(f"Evaluating segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {num_folds}-fold CV using test folds '{test_folds}'.")

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(num_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    # for test_fold in tqdm(test_folds):
    for test_fold in tqdm(test_folds):
        # Create dataframe.
        cols = {
            'fold': int,
            'dataset': str,
            'patient-id': str,
            'region': str,
            'metric': str,
            'value': float
        }
        df = pd.DataFrame(columns=cols.keys())

        # Build test loader.
        _, _, test_loader = Loader.build_loaders(datasets, region, num_folds=num_folds, test_fold=test_fold)

        # Add evaluations to dataframe.
        # for dataset_b, pat_id_b in tqdm(iter(test_loader), leave=False):
        for dataset_b, pat_id_b in tqdm(iter(test_loader), leave=False):
            if type(pat_id_b) == torch.Tensor:
                pat_id_b = pat_id_b.tolist()
            for dataset, pat_id in zip(dataset_b, pat_id_b):
                df = create_patient_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter, df=df)

        # Add fold.
        df['fold'] = test_fold

        # Set column types.
        df = df.astype(cols)

        # Save evaluation.
        filename = f'eval-folds-{num_folds}-test-{test_fold}'
        filepath = os.path.join(config.directories.evaluation, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)

def load_segmenter_evaluation(
    dataset: str,
    localiser: Tuple[str, str, str],
    segmenter: Tuple[str, str, str]) -> np.ndarray:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)
    set = ds.get(dataset, 'nifti')
    filepath = os.path.join(set.path, 'evaluation', 'segmenter', *localiser, *segmenter, 'eval.csv') 
    if not os.path.exists(filepath):
        raise ValueError(f"Segmenter evaluation for dataset '{set}', localiser '{localiser}' and segmenter '{segmenter}' not found.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def load_segmenter_evaluation_from_loader(
    datasets: Union[str, List[str]],
    localiser: types.ModelName,
    segmenter: types.ModelName,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> np.ndarray:
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)
    filename = f'eval-folds-{num_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluation, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
    if not os.path.exists(filepath):
        logging.error(f'filepath: {filepath}')
        raise ValueError(f"Segmenter evaluation for dataset '{datasets}', localiser '{localiser}', segmenter '{segmenter}', {num_folds}-fold CV with test fold {test_fold} not found.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def create_two_stage_evaluation_from_loader(
    datasets: Union[str, List[str]],
    region: str,
    localiser: types.ModelName,
    segmenter: types.ModelName,
    num_folds: Optional[int] = None,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    # Get unique name.
    localiser = Localiser.replace_checkpoint_aliases(*localiser)
    segmenter = Segmenter.replace_checkpoint_aliases(*segmenter)
    logging.info(f"Evaluating two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {num_folds}-fold CV using test folds '{test_folds}'.")

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(num_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]

    for test_fold in tqdm(test_folds):
        # Create dataframe.
        cols = {
            'fold': int,
            'patient-id': str,
            'region': str,
            'metric': str,
            'value': float
        }
        loc_df = pd.DataFrame(columns=cols.keys())
        seg_df = pd.DataFrame(columns=cols.keys())

        # Build test loader.
        _, _, test_loader = Loader.build_loaders(datasets, region, num_folds=num_folds, test_fold=test_fold)

        # Add evaluations to dataframe.
        for dataset_b, pat_id_b in tqdm(iter(test_loader), leave=False):
            if type(pat_id_b) == torch.Tensor:
                pat_id_b = pat_id_b.tolist()
            for dataset, pat_id in zip(dataset_b, pat_id_b):
                loc_df = create_patient_localiser_evaluation(dataset, pat_id, region, localiser, df=loc_df)
                seg_df = create_patient_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter, df=seg_df)

        # Add fold.
        loc_df['fold'] = test_fold
        seg_df['fold'] = test_fold

        # Set column types.
        loc_df = loc_df.astype(cols)
        seg_df = seg_df.astype(cols)

        # Save evaluations.
        filename = f'eval-folds-{num_folds}-test-{test_fold}'
        loc_filepath = os.path.join(config.directories.evaluation, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
        seg_filepath = os.path.join(config.directories.evaluation, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
        os.makedirs(os.path.dirname(loc_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(seg_filepath), exist_ok=True)
        loc_df.to_csv(loc_filepath, index=False)
        seg_df.to_csv(seg_filepath, index=False)
