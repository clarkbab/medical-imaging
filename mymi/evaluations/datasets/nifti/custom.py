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
from mymi.datasets import NiftiDataset
from mymi.geometry import get_box, centre_of_extent
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi.loaders import Loader, MultiLoader
from mymi.metrics import distances, dice, distances, extent_centre_distance, get_encaps_dist_mm
from mymi.models import replace_ckpt_alias
from mymi import logging
from mymi.predictions.datasets.nifti import get_institutional_localiser, load_multi_segmenter_prediction_dict, load_segmenter_predictions
from mymi.regions import RegionList, get_region_patch_size, get_region_tolerance, regions_to_list
from mymi.typing import ModelName, Region, Regions
from mymi.utils import append_row, arg_to_list, encode

def get_nnunet_multi_segmenter_evaluation(
    dataset: str,
    fold: int,
    region: Regions,
    pat_id: str,
    **kwargs) -> List[Dict[str, float]]:
    regions = regions_to_list(region)

    # Load predictions.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing
    filepath = f"/data/gpfs/projects/punim1413/mymi/datasets/nnunet/predictions/fold-{fold}/{pat_id}_processed.nii.gz"
    region_data = nib.load(filepath).get_fdata().astype(np.bool_)
    region_preds = {}
    for i, region in enumerate(regions):
        region_preds[region] = region_data[i + 1]
 
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

def get_nnunet_single_region_evaluation(
    dataset: str,
    fold: int,
    region: Region,
    pat_id: str,
    **kwargs) -> List[Dict[str, float]]:
    regions = regions_to_list(region)

    # Load predictions.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    spacing = pat.ct_spacing
    filepath = f"/data/gpfs/projects/punim1413/mymi/datasets/nnunet/predictions/single-region/{region}/fold-{fold}/{pat_id}_processed.nii.gz"
    region_data = nib.load(filepath).get_fdata().astype(np.bool_)
    region_preds = {}
    for i, region in enumerate(regions):
        region_preds[region] = region_data[i + 1]
 
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

def create_nnunet_single_region_evaluation(
    dataset: str,
    region: str,
    fold: int,
    exclude_like: Optional[str] = None,
    **kwargs) -> None:
    logging.arg_log('Evaluating multi-segmenter predictions (nnU-Net) for NIFTI dataset', ('dataset', 'fold'), (dataset, fold))

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

    # Add evaluations to dataframe.
    basepath = f"/data/gpfs/projects/punim1413/mymi/datasets/nnunet/predictions/single-region/{region}/fold-{fold}"
    files = list(sorted(os.listdir(basepath)))
    files = [f for f in files if f.endswith('_processed.nii.gz')]
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()
    for f in tqdm(files):
        f_pat_id = f.replace('_processed.nii.gz', '')
        if f_pat_id not in pat_ids:
            continue

        if exclude_like is not None:
            if exclude_like in f_pat_id:
                logging.info(f"Skipping '{dataset}:{f_pat_id}', matched 'exclude_like={exclude_like}'.")
                continue

        logging.info(f_pat_id)

        # Get metrics per region.
        region_metrics = get_nnunet_single_region_evaluation(dataset, fold, region, f_pat_id, **kwargs)
        for metrics in region_metrics:
            for metric, value in metrics.items():
                data = {
                    'fold': fold if fold is not None else np.nan,
                    'dataset': dataset,
                    'patient-id': f_pat_id,
                    'region': region,
                    'metric': metric,
                    'value': value
                }
                df = append_row(df, data)

    # Set column types.
    df = df.astype(cols)

    # Save evaluation.
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', 'nnunet', 'single-region', region, f'fold-{fold}', 'eval.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_nnunet_evaluation(
    dataset: str,
    fold: int) -> Union[np.ndarray, bool]:
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', 'nnunet', f'fold-{fold}', 'eval.csv')
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    return df

def load_nnunet_single_region_evaluation(
    dataset: str,
    region: str,
    fold: int) -> Union[np.ndarray, bool]:
    filepath = os.path.join(config.directories.evaluations, 'multi-segmenter', 'nnunet', 'single-region', region, f'fold-{fold}', 'eval.csv')
    df = pd.read_csv(filepath, dtype={'patient-id': str})
    return df
