import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Literal, Optional

class DiceLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-6) -> None:
        super().__init__()
        self.__epsilon = epsilon

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        reduction: Literal['mean', 'sum'] = 'mean') -> float:
        """
        returns: the dice loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of one-hot-encoded labels.
        """
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")
        if label.shape != pred.shape:
            if label.shape != (pred.shape[0], *pred.shape[2:]):
                raise ValueError(f"DiceLoss expects label to be one-hot-encoded and match prediction shape, or categorical and match on all dimensions except channels.")

            # One-hot encode the label.
            label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
            label = F.one_hot(label, num_classes=2)
            label = label.movedim(-1, 1)
        if mask is not None:
            assert mask.shape == pred.shape[:2]
        if weights is not None:
            assert weights.shape == pred.shape[:2]
            if weights.sum() != 1:
                raise ValueError(f"Weights must sum to 1. Got '{weights}'.")
        assert reduction in ('mean', 'sum')

        # Flatten volumetric data.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute dice coefficient.
        intersection = (pred * label).sum(dim=2)
        denominator = (pred + label).sum(dim=2)
        dice = (2. * intersection + self.__epsilon) / (denominator + self.__epsilon)

        # Apply mask across channels.
        if mask is not None:
            mask = torch.Tensor(mask).to(dice.device)
            mask = mask.unsqueeze(0).repeat(dice.shape[0], 1, 1)
            dice = mask * dice

        # Apply weights across channels.
        if weights is not None:
            weights = torch.Tensor(weights).to(dice.device)
            weights = weights.unsqueeze(0).repeat(dice.shape[0], 1, 1)
            dice = weights * dice

        # Reduce across batch and channel dimensions.
        if reduction == 'mean':
            dice = dice.mean()
        elif reduction == 'sum':
            dice = dice.sum()

        # Convert dice metric to loss.
        loss = -dice

        return loss
