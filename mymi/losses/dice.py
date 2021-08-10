import torch
from torch import nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5, weights=None):
        """
        kwargs:
            epsilon: small value to ensure we don't get division by zero.
            weights: the class weights, must sum to one.
        """
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weights = weights

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> float:
        """
        returns: the dice loss.
        args:
            pred: the 4D batch of network predictions (probabilities).
            label: the 4D batch of binary labels.
        """
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")

        # 'torch.argmax' isn't differentiable, so convert label to one-hot encoding
        # and calculate dice per-class/channel.
        label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
        label = F.one_hot(label, num_classes=2)
        label = label.movedim(-1, 1)
        if label.shape != pred.shape:
            raise ValueError(f"DiceLoss expects label shape (after one-hot, dim=1) and prediction shape to be equal. Got '{label.shape}' and '{pred.shape}'.")

        # Flatten volumetric data.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute dice coefficient.
        intersection = (pred * label).sum(dim=2)
        denominator = (pred + label).sum(dim=2)
        dice = (2. * intersection).clamp(min=self.epsilon) / denominator.clamp(min=self.epsilon)

        # Convert dice coef. to dice loss (larger is worse).
        # For dice metric, larger values are worse.
        loss = -dice

        # Determine weights.
        if self.weights:
            if len(self.weights) != len(loss.shape[1]):
                raise ValueError(f"DiceLoss expects number of weights equal to number of label classes. Got '{len(self.weights)}' and '{loss.shape[1]}'.")
            weights = self.weights
        else:
            weights = torch.ones_like(loss) / loss.shape[1]

        # Get weighted average dice loss.
        loss = (weights * loss).sum(1)

        # Get average across samples in batch.
        loss = loss.mean()

        return loss
