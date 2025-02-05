import torch
from torch import nn
from typing import *

class TverskyLoss(nn.Module):
    def __init__(self,
        alpha: float = 0.5,
        beta: float = 0.5,
        epsilon: float = 1e-6) -> None:
        super(TverskyLoss, self).__init__()
        self.__alpha = alpha
        self.__beta = beta
        self.__epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduce_channels: bool = False,
        weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        returns: the Tversky loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of binary labels.
            mask: the B x C batch of channel masks indicating which channels are present in GT.
            weights: the B x C batch of channel weights to weight the channel-wise loss components.
        """
        if pred.shape != label.shape:
            raise ValueError(f"Expected pred shape ({pred.shape}) to equal label shape ({label.shape}).")
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")

        # Flatten volumetric dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute TP, FP and FN.
        TP = (pred * label).sum(dim=2)
        FP = (pred * (1 - label)).sum(dim=2)
        FN = ((1 - pred) * label).sum(dim=2)

        # Calculate Tversky loss.
        tversky = (TP + self.__epsilon) / (TP + self.__alpha * FN + self.__beta * FP + self.__epsilon)

        # Apply mask/weights across channels.
        if mask is not None:
            tversky = mask * tversky
        if weights is not None:
            tversky = weights * tversky

        # Remove background channel.
        # When all foreground classes are present, background classes can be derived from
        # these. Otherwise, background mask is empty - so it's kind of useless.
        tversky = tversky[:, 1:]

        # Calculate loss.
        # Reduce across batch dimension only - preserve channel loss so we can see
        # how well individual channels are performing during training.
        loss = -tversky.mean(0)

        return loss
