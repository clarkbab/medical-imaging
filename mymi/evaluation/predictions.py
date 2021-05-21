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
        pred_dataset: the dataset of prediction labels.
        gt_dataset: the dataset of ground truth labels.
    kwargs:
        clear_cache: force the cache to clear.
        filter_errors: filter out patients with known errors. 
        pred_ct_from: the CT data matching the prediction labels.
    """
    # Create ground truth dataset.
    gt_ds = DicomDataset(gt_dataset)

    # Load evaluation patients.
    pred_ds = DicomDataset(pred_dataset, ct_from=pred_ct_from)
    pats = pred_ds.list_patients()

    # Load up labels.
    label_map = gt_ds.label_map(dataset=pred_dataset)
    internal_labels = list(label_map.internal)
    gt_labels = list(label_map[gt_dataset])
    pred_labels = list(label_map[pred_dataset])

    # Create dataframe.
    cols = {
        'pat-id': str,
        'metric': str
    }
    for label in internal_labels:
        cols[label] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add metrics for patients.
    for pat in pats:
        # Load ground truth.
        try:
            gt_label_data = gt_ds.patient(pat).label_data(clear_cache=clear_cache, labels=gt_labels)
        except ValueError as e:
            if filter_errors:
                logging.error(f"Error occurred while calling 'label_data' for dataset '{gt_ds.name}', patient '{pat}'.")
                logging.error(f"Error message: {e}")
                continue
            else:
                raise e

        # Load prediction data.
        pred_label_data = pred_ds.patient(pat).label_data(clear_cache=clear_cache, labels=pred_labels)

        # Add metrics for each label.
        dice_data = {
            'pat-id': pat,
            'metric': 'dice'
        }
        for internal_label, gt_label, pred_label in zip(internal_labels, gt_labels, pred_labels):
            if gt_label in gt_label_data and pred_label in pred_label_data:
                # Add dice.
                dice_score = dice(pred_label_data[pred_label], gt_label_data[gt_label])
                dice_data[internal_label] = dice_score
            else:
                continue

        # Add row.
        df = df.append(dice_data, ignore_index=True)

    return df
