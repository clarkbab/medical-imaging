import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import *

from mymi import config
from mymi.datasets import NiftiDataset
from mymi.loaders import get_holdout_split
from mymi.metrics import dice, distances, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_ckpt_alias
from mymi.predictions.nifti import load_segmenter_predictions
from mymi import logging
from mymi.regions import RegionList, get_region_patch_size, get_region_tolerance, regions_to_list
from mymi.typing import *
from mymi.utils import append_row, arg_to_list, encode, load_files_csv, save_csv

def get_segmenter_patient_evaluation(
    dataset: str,
    pat_id: str,
    model: str,
    **kwargs) -> List[Tuple[str, str, float]]:
    # Load pred/label.
    pred_data = load_segmenter_predictions(dataset, pat_id, model, **kwargs)
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing
    region_data = pat.region_data(**kwargs)

    metrics = []
    for r, pred in pred_data.items():
        label = region_data[r]

        # Dice.
        metrics.append((r, 'dice', dice(pred, label)))

        # Distances.
        if pred.sum() == 0 or label.sum() == 0:
            metrics.append((r, 'apl', np.nan))
            metrics.append((r, 'hd', np.nan))
            metrics.append((r, 'hd-95', np.nan))
            metrics.append((r, 'msd', np.nan))
            metrics.append((r, 'surface-dice', np.nan))
        else:
            # Calculate distances for OAR tolerance.
            tols = [0, 0.5, 1, 1.5, 2, 2.5]
            tol = get_region_tolerance(r)
            if tol is not None:
                tols.append(tol)
            dists = distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics.append((r, metric, value))

    return metrics

def create_all_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: Regions,
    model: ModelName,
    **kwargs) -> None:
    logging.arg_log('Creating multi-segmenter evaluation', ('dataset', 'region', 'model'), (dataset, region, model))
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    model = replace_ckpt_alias(model)

    # Create dataframe.
    cols = {
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Load patients.
    for dataset in datasets:
        set = NiftiDataset(dataset)
        pat_ids = set.list_patients()

        for pat_id in tqdm(pat_ids):
            # Get metrics per region.
            region_metrics = get_multi_segmenter_evaluation(dataset, regions, pat_id, model, **kwargs)
            for region, metrics in zip(regions, region_metrics):
                for metric, value in metrics.items():
                    data = {
                        'dataset': dataset,
                        'patient-id': pat_id,
                        'region': region,
                        'metric': metric,
                        'value': value
                    }
                    df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', *model, encode(datasets), encode(regions), 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
def create_replan_evaluation(
    dataset: Union[str, List[str]],
    region: Regions,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    # 'regions' is used to determine which patients are loaded (those that have at least one of
    # the listed regions).
    regions = regions_to_list(region)
    logging.arg_log('Evaluating mid-treatment against pre-treatment labels for NIFTI dataset', ('dataset', 'region', 'model'), (dataset, region, model))

    # Create dataframe.
    cols = {
        'fold': float,
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())


    # Load patients.
    for dataset in tqdm(datasets):
        set = NiftiDataset(dataset)
        pat_ids = set.list_patients()

        for pat_id in tqdm(pat_ids, leave=False):
            logging.info(f"Evaluating '{dataset}:{pat_id}'.")

            # Get metrics per region.
            region_metrics = get_replan_evaluation(dataset, regions, pat_id, model, **kwargs)
            for region, metrics in zip(regions, region_metrics):
                for metric, value in metrics.items():
                    data = {
                        'fold': test_fold if test_fold is not None else np.nan,
                        'dataset': dataset,
                        'patient-id': pat_id,
                        'region': region,
                        'metric': metric,
                        'value': value
                    }
                    df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    params = {
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter-replan', encode(datasets), encode(regions), encode(params), 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
def create_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: Regions,
    model: ModelName,
    exclude_like: Optional[str] = None,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    # 'regions' is used to determine which patients are loaded (those that have at least one of
    # the listed regions).
    model = replace_ckpt_alias(model)
    regions = regions_to_list(region)
    test_fold = kwargs.get('test_fold', None)
    logging.arg_log('Evaluating multi-segmenter predictions for NIFTI dataset', ('dataset', 'region', 'model'), (dataset, region, model))

    # Create dataframe.
    cols = {
        'fold': float,
        'dataset': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Build test loader.
    _, _, test_loader = MultiLoader.build_loaders(datasets, region=regions, **kwargs) 

    # Add evaluations to dataframe.
    test_loader = list(iter(test_loader))
    for pat_desc_b in tqdm(test_loader):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            logging.info(f"Evaluating '{dataset}:{pat_id}'.")

            if exclude_like is not None:
                if exclude_like in pat_id:
                    logging.info(f"Skipping '{dataset}:{pat_id}', matched 'exclude_like={exclude_like}'.")
                    continue

            # Get metrics per region.
            region_metrics = get_multi_segmenter_evaluation(dataset, regions, pat_id, model, **kwargs)
            for region, metrics in zip(regions, region_metrics):
                for metric, value in metrics.items():
                    data = {
                        'fold': test_fold if test_fold is not None else np.nan,
                        'dataset': dataset,
                        'patient-id': pat_id,
                        'region': region,
                        'metric': metric,
                        'value': value
                    }
                    df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    params = {
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_segmenter_holdout_evaluation(
    dataset: str,
    model: str,
    **kwargs) -> None:
    logging.arg_log('Evaluating segmenter predictions for NIFTI dataset', ('dataset', 'model'), (dataset, model))

    # Create dataframe.
    cols = {
        'dataset': str,
        'split': str,
        'patient-id': str,
        'region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Get patient split.
    _, _, tst = get_holdout_split(dataset, **kwargs)

    for pat_id in tqdm(tst):
        metrics = get_segmenter_patient_evaluation(dataset, pat_id, model, **kwargs)
        for region, metric, value in metrics:
            data = {
                'dataset': dataset,
                'split': 'test',
                'patient-id': pat_id,
                'region': region,
                'metric': metric,
                'value': value,
            }
            df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'evaluations', 'segmentations', f'{model}.csv')
    save_csv(df, filepath, overwrite=True)

def load_segmenter_holdout_evaluation(
    dataset: str,
    model: str,
    exists_only: bool = False,
    **kwargs) -> Union[np.ndarray, bool]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'evaluations', 'segmentations', f'{model}.csv')
    if not os.path.exists(filepath):
        if exists_only:
            return False
        else:
            raise ValueError(f"Segmenter evaluation for NIFTI dataset '{dataset}', model '{model}' not found.")
    elif exists_only:
        return True

    df = load_files_csv(filepath)
    if 'model' not in df.columns:
        df.insert(0, 'model', model)

    return df
