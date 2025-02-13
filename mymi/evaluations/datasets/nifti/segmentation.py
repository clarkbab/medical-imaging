import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm
from typing import *

from mymi import config
from mymi.datasets import NiftiDataset
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi.loaders import get_holdout_split
from mymi.metrics import dice, distances, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_ckpt_alias
from mymi.predictions.datasets.nifti import load_segmenter_predictions
from mymi import logging
from mymi.regions import RegionList, get_region_patch_size, get_region_tolerance, regions_to_list
from mymi.typing import *
from mymi.utils import append_row, arg_to_list, encode, load_csv, save_csv

def get_multi_segmenter_heatmap_evaluation(
    dataset: str,
    pat_id: str,
    model: ModelName,
    target_region: PatientRegions,
    layer: Union[str, List[str]],
    aux_region: PatientRegions) -> List[List[Tuple[Dict[str, float], List[Dict[str, float]]]]]:
    aux_regions = regions_to_list(aux_region)

    # Load region data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    region_data = pat.region_data(region=aux_region, regions_ignore_missing=True)

    # Process each target region.
    target_regions = regions_to_list(target_region)
    target_region_metrics = []
    for target_region in target_regions:
        # Load heatmap.
        heatmaps_full = load_multi_segmenter_heatmap(dataset, pat_id, model, target_region, layer)
        layers = arg_to_list(layer, str)
        if len(layers) == 1:
            heatmaps_full = [heatmaps_full]

        # Process each layer.
        layer_metrics = []
        for layer, heatmap_full in zip(layers, heatmaps_full):
            # Remove '-1' values - added to ensure heatmap is same size as CT but shoudln't be used for metric calculation.
            heatmap = heatmap_full[heatmap_full >= 0]

            # Calculate sum/max activation.
            layer_metrics_global = {}
            layer_metrics_global['max-act'] = heatmap.max()
            layer_metrics_global['sum-act'] = heatmap.sum()

            # Process each auxiliary region.
            aux_region_metrics = []
            for aux_region in aux_regions:
                if aux_region not in region_data:
                    continue
                
                # Load label - only evaluate label where heatmap != -1.
                label = region_data[aux_region]
                label = label[heatmap_full >= 0]
                dist = heatmap[label]

                # Calculate metrics.
                metric_names = [
                    'max-act',
                    'n-voxels',
                    'sum-act'
                ]
                metric_values = [
                    dist.max(),
                    int(label.sum()),
                    dist.sum()
                ]
                metrics = {}
                for name, value in zip(metric_names, metric_values):
                    metrics[name] = value

                aux_region_metrics.append(metrics)

            # Combine layer global metrics (e.g. max activation across entire heatmap) with
            # auxiliary region metrics (e.g. max activation in a specific region).
            combined_metrics = (layer_metrics_global, aux_region_metrics)
            layer_metrics.append(combined_metrics)
        target_region_metrics.append(layer_metrics)

    return target_region_metrics
    
def get_multi_segmenter_evaluation(
    dataset: str,
    region: PatientRegions,
    pat_id: str,
    model: ModelName,
    **kwargs) -> List[Dict[str, float]]:
    regions = arg_to_list(region, str)

    # Load predictions.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing
    region_preds = load_multi_segmenter_prediction_dict(dataset, pat_id, model, regions, **kwargs)
 
    region_metrics = []
    for region, pred in region_preds.items():
        # Patient ground truth may not have all the predicted regions.
        if not pat.has_regions(region):
            region_metrics.append({})
            continue
        
        # Load label.
        label = pat.region_data(region=region)[region]

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
        metrics = {}
        metrics['dice'] = dice(pred, label)

        # Distances.
        if pred.sum() == 0 or label.sum() == 0:
            metrics['apl'] = np.nan
            metrics['hd'] = np.nan
            metrics['hd-95'] = np.nan
            metrics['msd'] = np.nan
            metrics['surface-dice'] = np.nan
        else:
            # Calculate distances for OAR tolerance.
            tols = [0, 0.5, 1, 1.5, 2, 2.5]
            tol = get_region_tolerance(region)
            if tol is not None:
                tols.append(tol)
            dists = distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[metric] = value

            # Add 'deepmind' comparison.
            dists = distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[f'dm-{metric}'] = value

        region_metrics.append(metrics)

    return region_metrics

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
    
def create_multi_segmenter_heatmap_evaluation(
    dataset: Union[str, List[str]],
    model: ModelName,
    model_region: PatientRegions,
    target_region: PatientRegions,
    layer: Union[str, List[str]],
    aux_region: PatientRegions,
    exclude_like: Optional[str] = None,
    load_all_samples: bool = False,
    loader_shuffle_samples: bool = True,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_grouping: bool = False,
    use_loader_split_file: bool = False) -> None:
    datasets = arg_to_list(dataset, str)
    logging.arg_log('Evaluating multi-segmenter heatmaps for NIFTI dataset', ('dataset', 'model', 'model_region', 'target_region', 'layer', 'aux_region'), (dataset, model, model_region, target_region, layer, aux_region))
    model = replace_ckpt_alias(model)
    target_regions = regions_to_list(target_region)
    layers = arg_to_list(layer, str)
    aux_regions = regions_to_list(aux_region)

    # Create dataframe.
    cols = {
        'fold': float,
        'dataset': str,
        'patient-id': str,
        'target-region': str,
        'layer': str,
        'aux-region': str,
        'metric': str,
        'value': float
    }
    df = pd.DataFrame(columns=cols.keys())

    # Build test loader.
    _, _, test_loader = MultiLoader.build_loaders(datasets, load_all_samples=load_all_samples, n_folds=n_folds, region=model_region, shuffle_samples=loader_shuffle_samples, test_fold=test_fold, use_grouping=use_loader_grouping, use_split_file=use_loader_split_file) 

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
            target_region_metrics = get_multi_segmenter_heatmap_evaluation(dataset, pat_id, model, target_regions, layers, aux_regions)

            # Filter out aux_regions if patient doesn't have them.
            pat = NiftiDataset(dataset).patient(pat_id)
            aux_regions_pat = pat.list_regions(regions=aux_regions)

            for target_region, layer_metrics in zip(target_regions, target_region_metrics):
                for layer, (layer_global_metrics, aux_region_metrics) in zip(layers, layer_metrics):
                    # Add global metrics (e.g. max activation across entire heatmap).
                    for metric, value in layer_global_metrics.items():
                        data = {
                            'fold': test_fold if test_fold is not None else np.nan,
                            'dataset': dataset,
                            'patient-id': pat_id,
                            'target-region': target_region,
                            'layer': layer,
                            'aux-region': None,
                            'metric': metric,
                            'value': value
                        }
                        df = append_row(df, data)

                    for aux_region, metrics in zip(aux_regions_pat, aux_region_metrics):
                        for metric, value in metrics.items():
                            data = {
                                'fold': test_fold if test_fold is not None else np.nan,
                                'dataset': dataset,
                                'patient-id': pat_id,
                                'target-region': target_region,
                                'layer': layer,
                                'aux-region': aux_region,
                                'metric': metric,
                                'value': value
                            }
                            df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filename = f'folds-{n_folds}-test-{test_fold}-use-loader-split-file-{use_loader_split_file}-load-all-samples-{load_all_samples}.csv'
    filepath = os.path.join(config.directories.evaluations, 'heatmaps', *model, encode(datasets), encode(target_regions), encode(layers), encode(aux_regions), filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_all_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
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
    region: PatientRegions,
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
    region: PatientRegions,
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

    df = load_csv(filepath)
    if 'model' not in df.columns:
        df.insert(0, 'model', model)

    return df
