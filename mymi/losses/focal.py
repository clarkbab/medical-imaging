import torch
from torch import nn
from torch.nn import functional as F
from typing import Literal, Optional

class FocalLoss(nn.Module):
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
        returns: the Focal loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the the B x C x X x Y x Z batch of binary labels.
        """
        if label.shape != pred.shape:
            if label.shape != (pred.shape[0], *pred.shape[2:]):
                raise ValueError(f"FocalLoss expects label to be one-hot-encoded and match prediction shape, or categorical and match on all dimensions except channels.")

            # One-hot encode the label.
            label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
            label = F.one_hot(label, num_classes=2)
            label = label.movedim(-1, 1)
        if mask is not None:
            assert mask.shape == pred.shape[:2]
        if weights is not None:
            assert weights.shape == pred.shape[:2]
            weight_sum = weights.sum().round(decimals=3)
            if weight_sum != 1:
                raise ValueError(f"Weights must sum to 1. Got '{weight_sum}' (weights={weights}).")
        assert reduction in ('mean', 'sum')

        # Flatten spatial dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Calculate focal loss by reducing spatial dimension values.
        if pred[pred < 0].numel() != 0:
            print(pred[pred < 0])
        inner_loss = -((1 - pred) ** 2) * torch.log(pred + self.__epsilon) * label
        inner_loss = inner_loss.sum(axis=2)

        # Apply mask across channels.
        if mask is not None:
            mask = torch.Tensor(mask).to(inner_loss.device)
            mask = mask.unsqueeze(0).repeat(inner_loss.shape[0], 1, 1)
            inner_loss = mask * inner_loss

        # Apply weights across channels.
        if weights is not None:
            weights = torch.Tensor(weights).to(inner_loss.device)
            weights = weights.unsqueeze(0).repeat(inner_loss.shape[0], 1, 1)
            inner_loss = weights * inner_loss

        # Reduce over batch and channel dimensions.
        if reduction == 'mean':
            inner_loss = inner_loss.mean()
        elif reduction == 'sum':
            inner_loss = inner_loss.sum()

        # Normalise on number of voxels.
        if label.shape[2] == 0:
            print(label.shape)
        loss = (1 / label.shape[2] ) * inner_loss

        return loss
