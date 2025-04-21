import torch
from typing import *

from mymi import logging

class MSELoss(torch.nn.Module):
    def __init__(
        self,
        pad_threshold: Optional[float] = None) -> None:
        super().__init__()
        logging.info(f"Initialising MSE loss with pad_threshold={pad_threshold}.")
        self.__pad_threshold = pad_threshold

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> torch.Tensor:
        if self.__pad_threshold is not None:
            # Select subset of voxels that are both not padding. I.e. padding should not be involved
            # in the similarity calculation.
            indices = torch.argwhere((pred >= self.__pad_threshold) & (label >= self.__pad_threshold))
            indices = indices.unbind(dim=-1)
            pred = pred[indices]
            label = label[indices]
        mse = ((pred - label) ** 2).mean()
        return mse
