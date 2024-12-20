import numpy as np
import os
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

from mymi import dataset as ds
from mymi import logging
from mymi.models.systems import Localiser
from mymi import types

from ..prediction import get_localiser_prediction

def get_sample_localiser_prediction(
    dataset: str,
    sample_idx: str,
    localiser: pl.LightningModule,
    loc_size: types.ImageSize3D,
    loc_spacing: types.ImageSpacing3D,
    device: Optional[torch.device] = None) -> None:
    # Load data.
    set = ds.get(dataset, 'training')
    sample = set.sample(sample_idx)
    input = sample.input
    spacing = sample.spacing

    # Make prediction.
    pred = get_localiser_prediction(localiser, loc_size, loc_spacing, input, spacing, device=device)

    return pred
