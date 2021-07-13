from mymi.metrics.hausdorff_distance import sitk_hausdorff_distance
import pandas as pd
from typing import *

from mymi import cache
from mymi.dataset import DicomDataset
from mymi import logging
from mymi.metrics import dice, sitk_hausdorff_distance

@cache.function
def evaluate_predictions(
    pred_dataset: str,
    gt_dataset: str,
    clear_cache: bool = False,
    pred_ct_from: str = None) -> pd.DataFrame:
    """
    returns: a table of prediction results.
    args:
        pred_dataset: the dataset of prediction regions.
        gt_dataset: the dataset of ground truth regions.
    kwargs:
        clear_cache: force the cache to clear.
        pred_ct_from: the CT data matching the prediction regions.
    """
    # Create ground truth dataset.
    gt_ds = DicomDataset(gt_dataset)

    # Load evaluation patients.
    pred_ds = DicomDataset(pred_dataset, ct_from=pred_ct_from)
    pats = pred_ds.list_patients()

    # Load up regions.
    region_map = gt_ds.region_map(dataset=pred_dataset)
    internal_regions = list(region_map.internal)
    gt_regions = list(region_map[gt_dataset])
    pred_regions = list(region_map[pred_dataset])

    # Create dataframe.
    cols = {
        'patient-id': str,
        'metric': str
    }
    for region in internal_regions:
        cols[region] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add metrics for patients.
    for pat in pats:
        # Load patient spacing.
        spacing = gt_ds.patient(pat).ct_spacing(clear_cache=clear_cache)

        # Load ground truth.
        gt_region_data = gt_ds.patient(pat).region_data(clear_cache=clear_cache, regions=gt_regions)

        # Load prediction data.
        pred_region_data = pred_ds.patient(pat).region_data(clear_cache=clear_cache, regions=pred_regions)

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
        for i in range(len(internal_regions)):
            # Skip region if not contoured in both prediction and ground truth.
            if not (gt_regions[i] in gt_region_data and pred_regions[i] in pred_region_data):
                continue

            # Add dice.
            dice_score = dice(pred_region_data[pred_regions[i]], gt_region_data[gt_regions[i]])
            dice_data[internal_regions[i]] = dice_score

            # Add Hausdorff distance.
            hd_score = sitk_hausdorff_distance(pred_region_data[pred_regions[i]], gt_region_data[gt_regions[i]], spacing)
            hd_data[internal_regions[i]] = hd_score

        # Add rows.
        df = df.append(dice_data, ignore_index=True)
        df = df.append(hd_data, ignore_index=True)

    return df
