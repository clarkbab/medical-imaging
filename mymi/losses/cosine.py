from dicomset.utils import assert_shapes_equal
import numpy as np
import torch
from typing import *

from mymi import logging

class CosineLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logging.info(f"Initialising Cosine loss.")

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        ) -> torch.Tensor:
        assert_shapes_equal(pred, label)
        return (1 - torch.cos(2 * np.pi * (pred - label))).mean()
