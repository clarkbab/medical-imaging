from mymi.transforms.crop import crop_foreground_3D
import nibabel as nib
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
import torch
from tqdm import tqdm
from typing import Dict, List, Literal, Optional, Tuple, Union

from mymi import config
from mymi.dataset import NiftiDataset
from mymi.geometry import get_box, get_extent_centre
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi.loaders import AdaptiveLoader, Loader, MultiLoader
from mymi.metrics import all_distances, dice, distances_deepmind, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_ckpt_alias
from mymi import logging
from mymi.prediction.dataset.nifti.nifti import get_institutional_localiser, load_localiser_prediction, load_adaptive_segmenter_prediction_dict, load_adaptive_segmenter_no_oars_prediction_dict, load_multi_segmenter_prediction_dict, load_segmenter_prediction
from mymi.regions import RegionList, get_region_patch_size, get_region_tolerance, regions_to_list
from mymi.types import ModelName, PatientRegion, PatientRegions
from mymi.utils import append_row, arg_to_list, encode

def get_localiser_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_localiser_prediction(dataset, pat_id, localiser)
    set = NiftiDataset(dataset)
    label = set.patient(pat_id).region_data(region=region)[region].astype(np.bool_)

    # If 'SpinalCord' prediction extends further than ground truth in caudal z direction, then crop prediction.
    if region == 'SpinalCord':
        z_min_pred = np.nonzero(pred)[2].min()
        z_min_label = np.nonzero(label)[2].min()
        if z_min_pred < z_min_label:
            # Crop pred/label foreground voxels.
            crop = ((0, 0, z_min_label), label.shape)
            pred = crop_foreground_3D(pred, crop)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing
    if pred.sum() == 0 or label.sum() == 0:
        data['apl'] = np.nan
        data['hd'] = np.nan
        data['hd-95'] = np.nan
        data['msd'] = np.nan
        data['surface-dice'] = np.nan
    else:
        # Calculate distances for OAR tolerance.
        tols = [0, 0.5, 1, 1.5, 2, 2.5]
        tol = get_region_tolerance(region)
        if tol is not None:
            tols.append(tol)
        dists = all_distances(pred, label, spacing, tol=tols)
        for metric, value in dists.items():
            data[metric] = value

        # Add 'deepmind' comparison.
        dists = distances_deepmind(pred, label, spacing, tol=tols)
        for metric, value in dists.items():
            data[f'dm-{metric}'] = value

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
        # Get second-stage patch min/max coordinates.
        centre = get_extent_centre(pred)
        size = get_region_patch_size(region, spacing)
        min, max = get_box(centre, size)

        # Clip second-stage patch to label size - if necessary.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.clip(max, a_min=None, a_max=label.shape)

        # Convert second-stage patch coordinates into a label of ones so we can use 'get_encaps_dist_mm'.
        patch_label = np.zeros_like(label)
        slices = tuple([slice(l, h + 1) for l, h in zip(min, max)])
        patch_label[slices] = 1

        # Get extent distance.
        e_dist = get_encaps_dist_mm(patch_label, label, spacing)

    data['encaps-dist-mm-x'] = e_dist[0]
    data['encaps-dist-mm-y'] = e_dist[1]
    data['encaps-dist-mm-z'] = e_dist[2]

    return data

def create_localiser_evaluation_v2(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    logging.info(f"Evaluating localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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

    # Get patient IDs from original evaluation.
    # We have to evaluate the segmenter using the original evaluation patient IDs
    # as our 'Loader' now returns different patients per fold.
    orig_localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')
    orig_localiser = replace_ckpt_alias(orig_localiser)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    segmenter = (f'segmenter-{region}-v2', localiser[1], 'best')
    segmenter = replace_ckpt_alias(segmenter)
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *orig_localiser, *segmenter, encode(datasets), f'{filename}.csv')
    orig_df = pd.read_csv(filepath, dtype={'patient-id': str})
    orig_df = orig_df[['dataset', 'patient-id']].drop_duplicates()

    for i, row in tqdm(orig_df.iterrows()):
        dataset, pat_id = row['dataset'], row['patient-id']

        if region == 'BrachialPlexus_R' and dataset == 'PMCC-HN-TRAIN' and str(pat_id) == '177':
            # Skip this manually.
            continue

        metrics = get_localiser_evaluation(dataset, pat_id, region, localiser)
        for metric, value in metrics.items():
            data = {
                'fold': test_fold,
                'dataset': dataset,
                'patient-id': pat_id,
                'region': region,
                'metric': metric,
                'value': value
            }
            df = append_row(df, data)

    # Add fold.
    df['fold'] = test_fold

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_localiser_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    logging.info(f"Evaluating localiser predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Add evaluations to dataframe.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            metrics = get_localiser_evaluation(dataset, pat_id, region, localiser)
            for metric, value in metrics.items():
                data = {
                    'fold': test_fold,
                    'dataset': dataset,
                    'patient-id': pat_id,
                    'region': region,
                    'metric': metric,
                    'value': value
                }
                df = append_row(df, data)

    # Add fold.
    df['fold'] = test_fold

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_localiser_evaluation(
    datasets: Union[str, List[str]],
    localiser: ModelName,
    exists_only: bool = False,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> np.ndarray:
    localiser = replace_ckpt_alias(localiser)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'localiser', *localiser, encode(datasets), f'{filename}.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Localiser evaluation for dataset '{datasets}', localiser '{localiser}', {n_folds}-fold CV with test fold {test_fold} not found. Filepath: {filepath}.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def get_adaptive_segmenter_no_oars_evaluation(
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
    region_preds = load_adaptive_segmenter_no_oars_prediction_dict(dataset, pat_id, model, regions, **kwargs)
 
    region_metrics = []
    for region, pred in region_preds.items():
        # Patient ground truth may not have all the predicted regions.
        if not pat.has_regions(region):
            region_metrics.append({})
            continue
        
        # Load label.
        label = pat.region_data(region=region)[region]

        # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
        if region == 'SpinalCord' and pred.sum() != 0:
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
            dists = all_distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[metric] = value

            # Add 'deepmind' comparison.
            dists = distances_deepmind(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[f'dm-{metric}'] = value

        region_metrics.append(metrics)

    return region_metrics

def get_adaptive_segmenter_evaluation(
    dataset: str,
    region: PatientRegions,
    pat_id: str,
    model: ModelName) -> List[Dict[str, float]]:
    regions = arg_to_list(region, str)

    # Load predictions.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing
    region_preds = load_adaptive_segmenter_prediction_dict(dataset, pat_id, model, regions)
 
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
            dists = all_distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[metric] = value

            # Add 'deepmind' comparison.
            dists = distances_deepmind(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[f'dm-{metric}'] = value

        region_metrics.append(metrics)

    return region_metrics

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
            dists = all_distances(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[metric] = value

            # Add 'deepmind' comparison.
            dists = distances_deepmind(pred, label, spacing, tol=tols)
            for metric, value in dists.items():
                metrics[f'dm-{metric}'] = value

        region_metrics.append(metrics)

    return region_metrics

def get_segmenter_evaluation(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: ModelName,
    segmenter: ModelName) -> Dict[str, float]:
    # Get pred/ground truth.
    pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)
    set = NiftiDataset(dataset)
    label = set.patient(pat_id).region_data(region=region)[region].astype(np.bool_)

    # If 'SpinalCord' prediction extends further than ground truth in caudal z direction, then crop prediction.
    if region == 'SpinalCord':
        z_min_pred = np.nonzero(pred)[2].min()
        z_min_label = np.nonzero(label)[2].min()
        if z_min_pred < z_min_label:
            # Crop pred/label foreground voxels.
            crop = ((0, 0, z_min_label), label.shape)
            pred = crop_foreground_3D(pred, crop)

    # Dice.
    data = {}
    data['dice'] = dice(pred, label)

    # Distances.
    spacing = set.patient(pat_id).ct_spacing
    if pred.sum() == 0 or label.sum() == 0:
        data['apl'] = np.nan
        data['hd'] = np.nan
        data['hd-95'] = np.nan
        data['msd'] = np.nan
        data['surface-dice'] = np.nan
    else:
        # Calculate distances for OAR tolerance.
        tols = [0, 0.5, 1, 1.5, 2, 2.5]
        tol = get_region_tolerance(region)
        if tol is not None:
            tols.append(tol)
        dists = all_distances(pred, label, spacing, tol=tols)
        for metric, value in dists.items():
            data[metric] = value

        # Add 'deepmind' comparison.
        dists = distances_deepmind(pred, label, spacing, tol=tols)
        for metric, value in dists.items():
            data[f'dm-{metric}'] = value

    return data
    
def create_adaptive_segmenter_pt_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    # 'regions' is used to determine which patients are loaded (those that have at least one of
    # the listed regions).
    model = replace_ckpt_alias(model)
    regions = regions_to_list(region)
    test_fold = kwargs.get('test_fold', None)
    logging.arg_log('Evaluating adaptive segmenter predictions against PT ground truth for NIFTI dataset', ('dataset', 'region', 'model'), (dataset, region, model))

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
    _, _, test_loader = AdaptiveLoader.build_loaders(datasets, region=regions, **kwargs) 

    # Add evaluations to dataframe.
    test_loader = list(iter(test_loader))
    for pat_desc_b in tqdm(test_loader):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            logging.info(f"Evaluating '{dataset}:{pat_id}'.")

            # Skip pre-treatment patients.
            if '-0' in pat_id:
                logging.info(f"Skipping '{dataset}:{pat_id}'.")
                continue

            # Get metrics per region.
            region_metrics = get_adaptive_segmenter_pt_evaluation(dataset, regions, pat_id, model)
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
    filepath = os.path.join(config.directories.evaluations, 'adaptive-segmenter-pt', *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
def create_adaptive_segmenter_no_oars_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    include_ct: bool = True,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    # 'regions' is used to determine which patients are loaded (those that have at least one of
    # the listed regions).
    model = replace_ckpt_alias(model)
    regions = regions_to_list(region)
    test_fold = kwargs.get('test_fold', None)
    logging.arg_log(f"Evaluating adaptive segmenter predictions for NIFTI dataset (no prior OARs{ '' if include_ct else ' or CT' })", ('dataset', 'region', 'model'), (dataset, region, model))

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
    _, _, test_loader = AdaptiveLoader.build_loaders(datasets, region=regions, **kwargs) 

    # Add evaluations to dataframe.
    test_loader = list(iter(test_loader))
    for pat_desc_b in tqdm(test_loader):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            logging.info(f"Evaluating '{dataset}:{pat_id}'.")

            # Skip pre-treatment patients.
            if '-0' in pat_id:
                logging.info(f"Skipping '{dataset}:{pat_id}'.")
                continue

            # Get metrics per region.
            region_metrics = get_adaptive_segmenter_no_oars_evaluation(dataset, regions, pat_id, model, include_ct=include_ct, **kwargs)
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
    filepath = os.path.join(config.directories.evaluations, f"adaptive-segmenter-no-oars{ '' if include_ct else '-or-ct' }", *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
def create_adaptive_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    **kwargs) -> None:
    datasets = arg_to_list(dataset, str)
    # 'regions' is used to determine which patients are loaded (those that have at least one of
    # the listed regions).
    model = replace_ckpt_alias(model)
    regions = regions_to_list(region)
    test_fold = kwargs.get('test_fold', None)
    logging.arg_log('Evaluating adaptive segmenter predictions for NIFTI dataset', ('dataset', 'region', 'model'), (dataset, region, model))

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
    _, _, test_loader = AdaptiveLoader.build_loaders(datasets, region=regions, **kwargs) 

    # Add evaluations to dataframe.
    test_loader = list(iter(test_loader))
    for pat_desc_b in tqdm(test_loader):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            logging.info(f"Evaluating '{dataset}:{pat_id}'.")

            # Skip pre-treatment patients.
            if '-0' in pat_id:
                logging.info(f"Skipping '{dataset}:{pat_id}'.")
                continue

            # Get metrics per region.
            region_metrics = get_adaptive_segmenter_evaluation(dataset, regions, pat_id, model)
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
    filepath = os.path.join(config.directories.evaluations, 'adaptive-segmenter', *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    
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

def create_segmenter_evaluation_v2(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_train: float,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    logging.info(f"Evaluating segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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

    # Get patient IDs from original evaluation.
    # We have to evaluate the segmenter using the original evaluation patient IDs
    # as our 'Loader' now returns different patients per fold.
    orig_localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'best')
    orig_localiser = replace_ckpt_alias(orig_localiser)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *orig_localiser, *segmenter, encode(datasets), f'{filename}.csv')
    orig_df = pd.read_csv(filepath, dtype={'patient-id': str})
    orig_df = orig_df[['dataset', 'patient-id']].drop_duplicates()

    for i, row in tqdm(list(orig_df.iterrows())):
        dataset, pat_id = row['dataset'], row['patient-id']

        if region == 'BrachialPlexus_R' and dataset == 'PMCC-HN-TRAIN' and str(pat_id) == '177':
            # Skip this manually.
            continue

        metrics = get_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter)
        for metric, value in metrics.items():
            data = {
                'fold': test_fold,
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
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def create_segmenter_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    logging.info(f"Evaluating segmenter predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test fold '{test_fold}'.")

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
    _, _, test_loader = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=test_fold)

    # Add evaluations to dataframe.
    for pat_desc_b in tqdm(iter(test_loader)):
        if type(pat_desc_b) == torch.Tensor:
            pat_desc_b = pat_desc_b.tolist()
        for pat_desc in pat_desc_b:
            dataset, pat_id = pat_desc.split(':')
            metrics = get_segmenter_evaluation(dataset, pat_id, region, localiser, segmenter)
            for metric, value in metrics.items():
                data = {
                    'fold': test_fold,
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
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_adaptive_segmenter_pt_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    exists_only: bool = False,
    **kwargs) -> Union[np.ndarray, bool]:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    model = replace_ckpt_alias(model)
    params = {
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.evaluations, 'adaptive-segmenter-pt', *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Adaptive segmenter evaluation for dataset '{dataset}', model '{model}' not found. Params: {params}. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df

def load_adaptive_segmenter_no_oars_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    exists_only: bool = False,
    include_ct: bool = True,
    **kwargs) -> Union[np.ndarray, bool]:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    model = replace_ckpt_alias(model)
    params = {
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.evaluations, f"adaptive-segmenter-no-oars{ '' if include_ct else '-or-ct' }", *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Adaptive segmenter evaluation for dataset '{dataset}', model '{model}' not found. Params: {params}. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df

def load_adaptive_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    exists_only: bool = False,
    **kwargs) -> Union[np.ndarray, bool]:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    model = replace_ckpt_alias(model)
    params = {
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.evaluations, 'adaptive-segmenter', *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Adaptive segmenter evaluation for dataset '{dataset}', model '{model}' not found. Params: {params}. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df
    
def load_multi_segmenter_heatmap_evaluation(
    dataset: Union[str, List[str]],
    model: ModelName,
    target_region: PatientRegions,
    layer: Union[str, List[str]],
    aux_region: PatientRegions,
    exists_only: bool = False,
    load_all_samples: bool = False,
    loader_shuffle_samples: bool = False,
    n_folds: Optional[int] = None,
    test_fold: Optional[int] = None,
    use_loader_split_file: bool = False,
    use_loader_grouping: bool = False) -> Union[np.ndarray, bool]:
    datasets = arg_to_list(dataset, str)
    model = replace_ckpt_alias(model)
    target_regions = regions_to_list(target_region)
    layers = arg_to_list(layer, str)
    aux_regions = regions_to_list(aux_region)
    filename = f'folds-{n_folds}-test-{test_fold}-use-loader-split-file-{use_loader_split_file}-load-all-samples-{load_all_samples}.csv'
    filepath = os.path.join(config.directories.evaluations, 'heatmaps', *model, encode(datasets), encode(target_regions), encode(layers), encode(aux_regions), filename)
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Multi-segmenter heatmap evaluation for dataset '{dataset}', model '{model}', target_region '{target_region}' not found. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df

def load_all_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    exists_only: bool = False,
    **kwargs) -> DataFrame:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    model = replace_ckpt_alias(model)
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', *model, encode(datasets), encode(regions), 'eval.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Multi-segmenter (all) evaluation not found for model '{model}', dataset '{dataset}' and region '{region}'. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df

def load_multi_segmenter_evaluation(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model: ModelName,
    exists_only: bool = False,
    **kwargs) -> Union[np.ndarray, bool]:
    datasets = arg_to_list(dataset, str)
    regions = regions_to_list(region)
    model = replace_ckpt_alias(model)
    params = {
        'load_all_samples': kwargs.get('load_all_samples', False),
        'n_folds': kwargs.get('n_folds', None),
        'shuffle_samples': kwargs.get('shuffle_samples', True),
        'use_grouping': kwargs.get('use_grouping', False),
        'use_split_file': kwargs.get('use_split_file', False),
    }
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', *model, encode(datasets), encode(regions), encode(params), 'eval.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Multi-segmenter evaluation not found for model '{model}', dataset '{dataset}' and regions '{regions}'. Params: {params}. Filepath: {filepath}.")
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    df[['model-name', 'model-run', 'model-ckpt']] = model
    return df

def load_segmenter_evaluation(
    datasets: Union[str, List[str]],
    localiser: ModelName,
    segmenter: ModelName,
    exists_only: bool = False,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> Union[np.ndarray, bool]:
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    filename = f'eval-folds-{n_folds}-test-{test_fold}'
    filepath = os.path.join(config.directories.evaluations, 'segmenter', *localiser, *segmenter, encode(datasets), f'{filename}.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Segmenter evaluation for dataset '{datasets}', localiser '{localiser}', segmenter '{segmenter}', {n_folds}-fold CV with test fold {test_fold} not found. Filepath: {filepath}.")
    data = pd.read_csv(filepath, dtype={'patient-id': str})
    return data

def create_two_stage_evaluation(
    datasets: Union[str, List[str]],
    region: str,
    localiser: ModelName,
    segmenter: ModelName,
    n_folds: Optional[int] = 5,
    test_folds: Optional[Union[int, List[int], Literal['all']]] = None) -> None:
    # Get unique name.
    localiser = replace_ckpt_alias(localiser)
    segmenter = replace_ckpt_alias(segmenter)
    logging.info(f"Evaluating two-stage predictions for NIFTI datasets '{datasets}', region '{region}', localiser '{localiser}', segmenter '{segmenter}', with {n_folds}-fold CV using test folds '{test_folds}'.")

    # Perform for specified folds
    if test_folds == 'all':
        test_folds = list(range(n_folds))
    elif type(test_folds) == int:
        test_folds = [test_folds]
    else:
        raise ValueError(f"Invalid test_folds: {test_folds}, type ({type(test_folds)}).")

    for test_fold in tqdm(test_folds):
        create_localiser_evaluation(datasets, region, localiser, test_fold=test_fold)
        create_segmenter_evaluation(datasets, region, localiser, segmenter, test_fold=test_fold)
