from mymi.metrics.hausdorff_distance import sitk_hausdorff_distance
import pandas as pd
from typing import *

from mymi import cache
from mymi.dataset import DicomDataset
from mymi.metrics import dice, sitk_hausdorff_distance
from mymi import types

@cache.function
def evaluate_predictions(
    pred_dataset: str,
    gt_dataset: str,
    clear_cache: bool = False,
    pred_ct_from: str = None,
    raise_errors: bool = True,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    """
    returns: a table of prediction results.
    args:
        pred_dataset: the dataset of prediction regions.
        gt_dataset: the dataset of ground truth regions.
    kwargs:
        clear_cache: force the cache to clear.
        pred_ct_from: the CT data matching the prediction regions.
        raise_errors: raise known patient errors.
        regions: the prediction regions to evaluate.
    """
    # Load ground truth regions.
    gt_ds = DicomDataset(gt_dataset)
    gt_regions = gt_ds.region_names(clear_cache=clear_cache, raise_errors=raise_errors)

    # Load prediction patients/regions.
    pred_ds = DicomDataset(pred_dataset, ct_from=pred_ct_from)
    pats = pred_ds.list_patients()
    pred_regions = pred_ds.region_names(clear_cache=clear_cache, raise_errors=raise_errors)

    # Check that requested regions are present.
    if type(regions) == str:
        if regions == 'all':
            # Use all 'pred' regions.
            regions = pred_regions

            # Check that ground truth has required regions.
            for region in regions:
                if region not in gt_regions:
                    raise ValueError(f"Requested region '{region}' not found in dataset '{gt_dataset}'.")
        else:
            if regions not in gt_regions:
                raise ValueError(f"Requested region '{regions}' not found in dataset '{gt_dataset}'.")
            elif regions not in pred_regions:
                raise ValueError(f"Requested region '{regions}' not found in dataset '{pred_dataset}'.")
            
            # Convert to list.
            regions = [regions]
    else:
        for region in regions:
            if region not in gt_regions:
                raise ValueError(f"Requested region '{region}' not found in dataset '{gt_dataset}'.")
            elif region not in pred_regions:
                raise ValueError(f"Requested region '{region}' not found in dataset '{pred_dataset}'.")

    # Load patients.
    pats = pred_ds.list_patients()
    # TODO: filter patients with errors.

    # Create dataframe.
    cols = {
        'patient-id': str,
        'metric': str
    }
    for region in regions:
        cols[region] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add metrics for patients.
    for pat in pats:
        # Load patient spacing.
        spacing = gt_ds.patient(pat).ct_spacing(clear_cache=clear_cache)

        # Load ground truth.
        gt_region_data = gt_ds.patient(pat).region_data(clear_cache=clear_cache, regions=regions)

        # Load prediction data.
        pred_region_data = pred_ds.patient(pat).region_data(clear_cache=clear_cache, regions=regions)

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
        for i, region in enumerate(regions):
            # Add dice.
            dice_score = dice()
            dice_score = dice(pred_region_data[pred_regions[i]], gt_region_data[gt_regions[i]])
            dice_data[internal_regions[i]] = dice_score

            # Add Hausdorff distance.
            hd_score = sitk_hausdorff_distance(pred_region_data[pred_regions[i]], gt_region_data[gt_regions[i]], spacing)
            hd_data[internal_regions[i]] = hd_score

        # Add rows.
        df = df.append(dice_data, ignore_index=True)
        df = df.append(hd_data, ignore_index=True)

    return df
