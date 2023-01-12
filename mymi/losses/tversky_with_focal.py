import torch
from torch import nn

from .focal import FocalLoss
from .tversky import TverskyLoss

class TverskyWithFocalLoss(nn.Module):
    def __init__(self,
        alpha: float = 0.5,
        beta: float = 0.5,
        epsilon: float = 1e-6,
        lam: float = 0.5) -> None:
        super(TverskyWithFocalLoss, self).__init__()
        self.__tversky = TverskyLoss(alpha=alpha, beta=beta, epsilon=epsilon)
        self.__focal = FocalLoss()
        self.__lam = lam

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        class_mask: torch.Tensor,
        class_weights: torch.Tensor) -> float:
        """
        returns: the TverskyWithFocal loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the the B x C x X x Y x Z batch of binary labels.
        """
        if pred.shape != label.shape:
            raise ValueError(f"Expected pred shape ({pred.shape}) to equal label shape ({label.shape}).")
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")
        
        # Get hybrid loss.
        loss = self.__tversky(pred, label, class_mask, class_weights) + self.__lam * self.__focal(pred, label, class_mask, class_weights)

        return loss
