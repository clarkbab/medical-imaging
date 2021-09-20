import numpy as np
import pandas as pd
from tqdm import tqdm

from mymi import cache
from mymi import dataset as ds
from mymi.metrics import dice, hausdorff_distance
from mymi.models.systems import Localiser, Segmenter
from mymi import logging
from mymi.prediction import get_patient_segmentation
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

    # Pre-load model - so we don't need to reload for each prediction.
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
        pred = get_patient_segmentation(localiser, segmenter, set, pat, device=device)
        label = set.patient(pat).region_data()[region]

        # Add metrics.
        dsc_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'dsc'
        }
        hd_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'hd'
        }
        hd_avg_data = {
            'patient-id': pat,
            'region': region,
            'metric': 'hd-avg'
        }

        # DSC.
        dsc_score = dice(pred, label)
        dsc_data[region] = dsc_score

        # HD.
        spacing = set.patient(pat).ct_spacing()
        hd, hd_avg = hausdorff_distance(pred, label, spacing)
        hd_data[region] = hd
        hd_avg_data[region] = hd_avg

        df = df.append(dsc_data, ignore_index=True)
        df = df.append(hd_data, ignore_index=True)
        df = df.append(hd_avg_data, ignore_index=True)

    # Set index.
    df = df.set_index('patient-id')

    return df
