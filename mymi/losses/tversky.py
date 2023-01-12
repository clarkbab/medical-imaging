import torch
from torch import nn

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
        class_mask: torch.Tensor,
        class_weights: torch.Tensor) -> float:
        """
        returns: the Tversky loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of binary labels.
            class_mask: the B x C batch of class masks indicating which regions are present.
            class_weights: the B x C batch of class weights to up/down-weight particular regions during training.
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
        FP = (pred * ~label).sum(dim=2)
        FN = ((1 - pred) * label).sum(dim=2)

        # Calculate Tversky loss.
        n_classes = label.shape[1]
        inner_loss = (TP + self.__epsilon) / (TP + self.__alpha * FN + self.__beta * FP + self.__epsilon)
        inner_loss = class_mask * class_weights * inner_loss
        loss = n_classes - inner_loss.sum(axis=1)

        # Average across batch samples.
        loss = loss.mean()

        return loss
