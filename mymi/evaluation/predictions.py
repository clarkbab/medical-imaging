import pandas as pd
from typing import *

from mymi.cache import cached_function
from mymi.dataset import DicomDataset
from mymi import logging
from mymi.metrics import dice

@cached_function
def evaluate_predictions(
    pred_dataset: str,
    gt_dataset: str,
    clear_cache: bool = False,
    filter_errors: bool = False,
    pred_ct_from: str = None) -> pd.DataFrame:
    """
    returns: a table of prediction results.
    args:
        pred_dataset: the dataset of prediction regions.
        gt_dataset: the dataset of ground truth regions.
    kwargs:
        clear_cache: force the cache to clear.
        filter_errors: filter out patients with known errors. 
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
        # Load ground truth.
        try:
            gt_region_data = gt_ds.patient(pat).region_data(clear_cache=clear_cache, regions=gt_regions)
        except ValueError as e:
            if filter_errors:
                logging.error(f"Patient filtered due to error calling 'region_data' for dataset '{gt_ds.name}', patient '{pat}'.")
                logging.error(f"Error message: {e}")
                continue
            else:
                raise e

        # Load prediction data.
        pred_region_data = pred_ds.patient(pat).region_data(clear_cache=clear_cache, regions=pred_regions)

        # Add metrics for each region.
        dice_data = {
            'patient-id': pat,
            'metric': 'dice'
        }
        for internal_region, gt_region, pred_region in zip(internal_regions, gt_regions, pred_regions):
            if gt_region in gt_region_data and pred_region in pred_region_data:
                # Add dice.
                dice_score = dice(pred_region_data[pred_region], gt_region_data[gt_region])
                dice_data[internal_region] = dice_score
            else:
                continue

        # Add row.
        df = df.append(dice_data, ignore_index=True)

    return df
