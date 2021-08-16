import numpy as np
import torch

from mymi.dataset import Dataset
from mymi import types

from .localiser import get_patient_box
from .segmenter import get_patient_segmentation_patch

def get_patient_segmentation(
    dataset: Dataset,
    id: types.PatientID,
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
    box = get_patient_box(dataset, id, clear_cache=clear_cache, device=device)
    seg = get_patient_segmentation_patch(dataset, id, box, clear_cache=clear_cache, device=device)
    return seg
