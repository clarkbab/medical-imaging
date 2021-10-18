import numpy as np
import pandas as pd
from tqdm import tqdm

from mymi import cache
from mymi import dataset as ds
from mymi.metrics import dice, distances
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction import get_two_stage_prediction
from mymi import types

def evaluate_model(
    dataset: str,
    localiser: types.Model,
    segmenter: types.Model,
    region: str) -> pd.DataFrame:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Evaluating on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Evaluating on CPU...')

    # Load dataset.
    set = ds.get(dataset, 'dicom')
    pats = set.list_patients(regions=region)

    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    if type(segmenter) == tuple:
        segmenter = Segmenter.load(*segmenter)

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str,
        'metric': str
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Get pred/ground truth.
        pred = get_two(set, pat, localiser, segmenter, device=device)
        label = set.patient(pat).region_data()[region]

        # Add metrics.
        dsc_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'dice'
        }
        hd_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'hausdorff'
        }
        hd_avg_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'average-hausdorff'
        }
        sd_avg_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'average-surface'
        }
        sd_med_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'median-surface'
        }
        sd_std_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'std-surface'
        }
        sd_max_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'max-surface'
        }

        # Dice.
        dsc_score = dice(pred, label)
        dsc_data[region] = dsc_score
        df = df.append(dsc_data, ignore_index=True)

        # Hausdorff.
        spacing = set.patient(pat).ct_spacing()
        hd, hd_avg = hausdorff_distance(pred, label, spacing)
        hd_data[region] = hd
        hd_avg_data[region] = hd_avg
        df = df.append(hd_data, ignore_index=True)
        df = df.append(hd_avg_data, ignore_index=True)

        # Symmetric surface distance.
        sd_mean, sd_median, sd_std, sd_max = symmetric_surface_distance(pred, label, spacing)
        sd_mean_data[region] = sd_mean
        sd_median_data[region] = sd_median
        sd_std_data[region] = sd_std
        sd_max_data[region] = sd_max
        df = df.append(sd_mean, ignore_index=True)
        df = df.append(sd_median, ignore_index=True)
        df = df.append(sd_std, ignore_index=True)
        df = df.append(sd_max, ignore_index=True)

    # Set index.
    df = df.set_index('patient-id')

    return df
