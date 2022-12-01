import torch
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self) -> None:
        super(FocalLoss, self).__init__()

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        class_mask: torch.Tensor,
        class_weights: torch.Tensor) -> float:
        """
        returns: the Focal loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the the B x C x X x Y x Z batch of binary labels.
        """
        if pred.shape != label.shape:
            raise ValueError(f"Expected pred shape ({pred.shape}) to equal label shape ({label.shape}).")
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")

        # Flatten volumetric dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Calculate focal loss.
        n_voxels = label.shape[2]
        inner_loss = label * ((1 - pred) ** 2) * torch.log(pred)
        inner_loss = inner_loss.sum(axis=2)
        inner_loss = class_mask * class_weights * inner_loss
        loss = -(1 / n_voxels) * inner_loss.sum(axis=1)

        # Average over batch samples.
        loss = loss.mean()

        return loss
