import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from mymi import dataset as ds
from mymi import logging
from mymi.models.systems import Localiser
from mymi import types

def get_localiser_prediction(
    dataset: str,
    partition: str,
    sample_id: int,
    localiser: types.Model,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu'),
    logits: bool = False,
    return_seg: bool = False) -> Union[Optional[types.Box3D], Tuple[Optional[types.Box3D], np.ndarray]]:
    # Load model if not already loaded.
    if type(localiser) == tuple:
        localiser = Localiser.load(*localiser, logits=logits)
    localiser.eval()
    localiser.to(device)
    localiser_size = (128, 128, 96)
    localiser_spacing = (4, 4, 6.625)

    # Load the sample - assume the input has been processed to the correct size/spacing.
    set = ds.get(dataset, 'training')
    input = set.partition(partition).sample(sample_id).input()

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
    if pred.sum() == 0 or logits:
        box = None
    else:
        non_zero = np.argwhere(pred != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)

    # Create result.
    if return_seg:
        return (box, pred)
    else:
        return box

def create_localiser_predictions(
    dataset: str,
    partitions: Union[str, List[str]],
    localiser: Tuple[str, str, str],
    region: str,
    clear_cache: bool = False,
    logits: bool = False) -> None:
    logging.info(f"Predicting {'logits ' if logits else ''}for dataset '{dataset}', partitions '{partitions}', region '{region}' using localiser '{localiser}'.")

    # Load gpu if available.
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logging.info('Predicting on GPU...')
    else:
        device = torch.device('cpu')
        logging.info('Predicting on CPU...')

    # Convert partitions arg to list.
    if type(partitions) == str:
        partitions = [partitions]

    # Get dataset.
    set = ds.get(dataset, 'training')

    # Load model.
    localiser_args = localiser
    localiser = Localiser.load(*localiser, logits=logits)
    
    for partition in partitions:
        # Load patients.
        samples = set.partition(partition).list_samples(regions=region)

        for sample in tqdm(samples):
            # Make prediction.
            _, pred = get_localiser_prediction(dataset, partition, sample, localiser, clear_cache=clear_cache, device=device, logits=logits, return_seg=True)

            # Save in folder.
            basepath = os.path.join(set.path, 'predictions', partition, 'localiser', *localiser_args) 
            if logits:
                filepath = os.path.join(basepath, 'logits', f"{sample}.npz")
            else:
                filepath = os.path.join(basepath, f"{sample}.npz")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.savez(filepath, data=pred)

def load_localiser_prediction(
    dataset: str,
    partition: str,
    sample_id: int,
    localiser: Tuple[str, str, str],
    logits: bool = False,
    return_seg: bool = False) -> np.ndarray:
    set = ds.get(dataset, 'training')
    # Load segmentation.
    basepath = os.path.join(set.path, 'predictions', partition, 'localiser', *localiser) 
    if logits:
        filepath = os.path.join(basepath, 'logits', f"{sample_id}.npz")
    else:
        filepath = os.path.join(basepath, f"{sample_id}.npz")
    if not os.path.exists(filepath):
        raise ValueError(f"Prediction for dataset '{set}', localiser '{localiser}' not found.")
    pred = np.load(filepath)['data']

    # Get bounding box.
    if pred.sum() == 0 or logits:
        box = None
    else:
        non_zero = np.argwhere(pred != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        box = (min, max)

    if return_seg:
        return (box, pred)
    else:
        return box
