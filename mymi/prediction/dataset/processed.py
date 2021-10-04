import numpy as np
import os
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Union

from mymi import dataset as ds
from mymi import logging
from mymi.models.systems import Localiser
from mymi import types

def create_localiser_prediction(
    dataset: str,
    partition: str,
    sample_idx: int,
    localiser: types.Model,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    return_seg: bool = False) -> Union[Optional[types.Box3D], Tuple[Optional[types.Box3D], np.ndarray]]:
    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser)
    localiser.eval()
    localiser.to(device)
    localiser_size = (128, 128, 96)
    localiser_spacing = (4, 4, 6.625)

    # Load the sample - assume the sample has the spacing required by the localiser.
    set = ds.get(dataset, 'processed')
    input = set.partition(partition).sample(sample_idx).input()

    # Get localiser result.
    input = torch.Tensor(input)
    input = input.unsqueeze(0)      # Add 'batch' dimension.
    input = input.unsqueeze(1)      # Add 'channel' dimension.
    input = input.float()
    input = input.to(device)
    with torch.no_grad():
        pred = localiser(input)
    pred = pred.squeeze(0)          # Remove 'batch' dimension.

    # Get OAR extent.
    if pred.sum() > 0:
        non_zero = np.argwhere(pred != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)
    else:
        box = None

    # Create result.
    if return_seg:
        return (box, pred)
    else:
        return box

def create_localiser_predictions(
    dataset: str,
    partition: str,
    localiser: Tuple[str, str, str],
    region: str,
    clear_cache: bool = False) -> None:
    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Load patients.
    set = ds.get(dataset, 'processed')
    samples = set.partition(partition).list_samples(regions=region)

    # Load models.
    localiser_args = localiser
    localiser = Localiser.load(*localiser)

    for sample in tqdm(samples):
        # Make prediction.
        _, pred = create_localiser_prediction(dataset, partition, sample, localiser, clear_cache=clear_cache, device=device, return_seg=True)

        # Save in folder.
        filepath = os.path.join(set.path, 'predictions', partition, 'localiser', *localiser_args, f"{sample}.npz") 
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, data=pred)

def get_localiser_prediction(
    dataset: str,
    partition: str,
    sample_idx: int,
    localiser: Tuple[str, str, str]) -> np.ndarray:
    set = ds.get(dataset, 'processed')
    filepath = os.path.join(set.path, 'predictions', partition, 'localiser', *localiser, f"{sample_idx}.npz") 
    data = np.load(filepath)['data']
    return data
