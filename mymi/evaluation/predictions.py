from re import I
from mymi.metrics.hausdorff_distance import sitk_hausdorff_distance
import numpy as np
import pandas as pd
from typing import *
from tqdm import tqdm

from mymi import cache
from mymi.dataset import DicomDataset
from mymi.metrics import dice, sitk_hausdorff_distance
from mymi import logging
from mymi import types

# @cache.function
def evaluate_predictions(
    pred_dataset: str,
    gt_dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    """
    returns: a table of prediction results.
    args:
        pred_dataset: the dataset of prediction regions.
        gt_dataset: the dataset of ground truth regions.
    kwargs:
        clear_cache: force the cache to clear.
        regions: the requested prediction regions.
    """
    # Load patients to predict on.
    pred_ds = DicomDataset(pred_dataset)
    pats = pred_ds.list_patients()

    # Load GT dataset.
    gt_ds = DicomDataset(gt_dataset)

    # Convert regions to list.
    if type(regions) == str:
        if regions == 'all':
            regions = list(pred_ds.region_names(clear_cache=clear_cache).region.unique())
        else:
            regions = [regions]

    # Create dataframe.
    cols = {
        'patient-id': str,
        'metric': str
    }
    for region in regions:
        cols[region] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add metrics for patients.
    logging.info(f"Evaluating patient predictions..")
    for pat in tqdm(pats):
        # Filter patient if not present in GT.
        if not gt_ds.has_patient(pat):
            logging.info(f"Skipping patient '{pat}', not present in ground truth dataset.")
            continue

        # Get overlap between available pred/GT regions and requested regions.
        pred_regions = list(pred_ds.patient(pat).region_names(clear_cache=clear_cache).region)
        gt_regions = list(gt_ds.patient(pat).region_names(clear_cache=clear_cache).region)
        overlap_regions = np.intersect1d(np.intersect1d(pred_regions, gt_regions), regions) 

        # Filter if no overlapping regions.
        if len(overlap_regions) == 0:
            logging.info(f"Skipping patient '{pat}', no requested region is present in both prediction and ground truth datasets.")
            continue

        # Load prediction/GT data.
        pred_region_data = pred_ds.patient(pat).region_data(clear_cache=clear_cache, regions=overlap_regions)
        gt_region_data = gt_ds.patient(pat).region_data(clear_cache=clear_cache, regions=overlap_regions)

        # Create empty dataframe rows.
        dice_data = {
            'patient-id': pat,
            'metric': 'dice'
        }
        hd_data = {
            'patient-id': pat,
            'metric': 'hausdorff'
        }

        # Calculate metrics for each region.
        spacing = gt_ds.patient(pat).ct_spacing(clear_cache=clear_cache)
        for region in overlap_regions:
            # Add dice.
            dice_score = dice(pred_region_data[region], gt_region_data[region])
            dice_data[region] = dice_score

            # Add Hausdorff distance.
            hd_score = sitk_hausdorff_distance(pred_region_data[region], gt_region_data[region], spacing)
            hd_data[region] = hd_score

        # Add rows.
        df = df.append(dice_data, ignore_index=True)
        df = df.append(hd_data, ignore_index=True)

    return df
