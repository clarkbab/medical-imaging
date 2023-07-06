import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal, Optional

from .dice import DiceLoss
from .focal import FocalLoss

class DiceWithFocalLoss(nn.Module):
    def __init__(self,
        epsilon: float = 1e-6,
        lam: float = 0.5) -> None:
        super().__init__()
        self.__dice = DiceLoss(epsilon=epsilon)
        self.__focal = FocalLoss()
        self.__lam = lam

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        reduction: Literal['mean', 'sum'] = 'mean') -> float:
        """
        returns: the DiceWithFocal loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the the B x C x X x Y x Z batch of binary labels.
            mask: channels to exclude from calculation.
            weights: weights to apply to channels.
        """
        if label.shape != pred.shape:
            if label.shape != (pred.shape[0], *pred.shape[2:]):
                raise ValueError(f"DiceWithFocalLoss expects label to be one-hot-encoded and match prediction shape, or categorical and match on all dimensions except channels.")

            # One-hot encode the label.
            label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
            label = F.one_hot(label, num_classes=2)
            label = label.movedim(-1, 1)
            label = label.bool()
        if mask is not None:
            assert mask.shape == pred.shape[:2]
            if mask.shape != pred.shape[:2]:
                raise ValueError(f"Expected mask to have shape '{pred.shape[:2]}', got '{mask.shape}'.")
        batch_size = pred.shape[0]
        if weights is not None:
            if weights.shape != pred.shape[:2]:
                raise ValueError(f"Expected weights to have shape '{pred.shape[:2]}', got '{weights.shape}'.")
            for b in range(batch_size):
                weight_sum = weights[b].sum().round(decimals=3)
                if weight_sum != 1:
                    raise ValueError(f"Weights (batch={b}) must sum to 1. Got '{weight_sum}' (weights={weights[b]}).")
        
        # Get hybrid loss.
        loss = self.__dice(pred, label, mask=mask, weights=weights, reduction=reduction) + self.__lam * self.__focal(pred, label, mask=mask, weights=weights, reduction=reduction)

        return loss
