import numpy as np
import pytorch_lightning as pl
import torch

from mymi.dataset import Dataset
from mymi import types

from .localiser import get_localiser_prediction
from .segmenter import get_segmenter_prediction

def get_two_stage_prediction(
    dataset: Dataset,
    pat_id: types.PatientID,
    localiser: types.Model,
    segmenter: types.Model,
    clear_cache: bool = False,
    device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    returns: the patient segmentation.
    args:
        dataset: the dataset name.
        patient: the patient ID.
    kwargs:
        clear_cache: force the cache to clear.
        device: the device to perform network calcs on.
    """
    box = get_localiser_prediction(dataset, pat_id, localiser, clear_cache=clear_cache, device=device)
    seg = get_segmenter_prediction(dataset, pat_id, segmenter, box, clear_cache=clear_cache, device=device)
    return seg
