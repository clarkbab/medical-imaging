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

    def forward(self, pred, label):
        """
        returns: the Dice loss for the prediction and label.
        args:
            pred: the prediction logits.
            label: the output tensor.
        """
        # Convert logits to probabilities.
        pred = pred.softmax(dim=1)

        # 'torch.argmax' isn't differentiable, so convert label to one-hot encoding
        # and calculate dice per-class/channel.
        label = F.one_hot(label)
        label = label.movedim(-1, 1)
        assert pred.shape == label.shape, f"Prediction ({pred.shape}) and label ({label.shape}) must have the same shape."

        # Flatten data.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute dice coefficient.
        intersection = (pred * label).sum(dim=2)
        denominator = (pred + label).sum(dim=2)
        dice_coef = (2. * intersection).clamp(min=self.epsilon) / denominator.clamp(min=self.epsilon)

        # Convert dice coef. to dice loss (larger is worse).
        dice_loss = 1 - dice_coef

        # Determine weights.
        if self.weights is None:
            weights = torch.ones_like(dice_loss) / dice_loss.shape[1]
        else:
            assert len(dice_loss) == len(self.weights), f"Number of classes ({len(dice)}) should equal number of class weights ({len(self.weights)})."
            weights = self.weights 

        # Get weighted average dice loss.
        dice_loss = (weights * dice_loss).sum(1)

        # Get average across samples in batch.
        dice_loss = dice_loss.mean()

        return dice_loss
