import torch
from torch import nn

class NCCLoss(nn.Module):
    def __init__(self) -> None:
        # Delay tensorflow import.
        from monai.losses import LocalNormalizedCrossCorrelationLoss
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> float:
        """
        args:
            pred: the B x C x X x Y x Z batch of predicted images.
            label: the the B x C x X x Y x Z batch of ground truth images.
        """
        return LocalNormalizedCrossCorrelationLoss()(pred, label)
