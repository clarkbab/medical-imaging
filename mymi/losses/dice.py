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
        reduction: Literal['mean', 'sum'] = 'mean',
        sc_channel: Optional[int] = None) -> float:
        """
        returns: the dice loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of one-hot-encoded labels.
        """
        assert pred.shape[0] == 1   # TODO: 'SpinalCord' specific loss doesn't handle batch size > 1.
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")
        if pred.dtype == torch.bool:
            raise ValueError(f"Pred should have float type, not bool.")
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
            weight_sum = weights.sum().round(decimals=3)
            if weight_sum != 1:
                raise ValueError(f"Weights must sum to 1. Got '{weight_sum}' (weights={weights}).")
        assert reduction in ('mean', 'sum')

        # Separate 'SpinalCord' channel if necessary.
        if sc_channel is not None:
            # Split 'SpinalCord' channel out from main data.
            sc_pred = pred[:, sc_channel]
            sc_label = label[:, sc_channel]
            pred_a = pred[:, :sc_channel]
            pred_b = pred[:, (sc_channel + 1):]
            pred = torch.concat((pred_a, pred_b), dim=1)
            label_a = label[:, :sc_channel]
            label_b = label[:, (sc_channel + 1):]
            label = torch.concat((label_a, label_b), dim=1)

            # Remove all voxels below 'label_min_z' from loss calculation.
            # We don't want to penalise the model for predicting further in inferior
            # direction than the label.
            label_min_z = torch.argwhere(sc_label).min(axis=0).values[3].item()
            sc_pred = sc_pred[:, :, :, label_min_z:]
            sc_label = sc_label[:, :, :, label_min_z:]

        # Flatten volumetric data.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)
        if sc_channel is not None:
            sc_pred = sc_pred.flatten(start_dim=1)
            sc_label = sc_label.flatten(start_dim=1)

        # Compute dice coefficient.
        intersection = (pred * label).sum(dim=2)
        denominator = (pred + label).sum(dim=2)
        dice = (2. * intersection + self.__epsilon) / (denominator + self.__epsilon)
        if sc_channel is not None:
            # Calculate 'SpinalCord' dice.
            sc_intersection = (sc_pred * sc_label).sum(dim=1)
            sc_denominator = (sc_pred + sc_label).sum(dim=1)
            sc_dice = (2. * sc_intersection + self.__epsilon) / (sc_denominator + self.__epsilon)

            # Insert result back into main dice results.
            dice_a = dice[:, :sc_channel]
            sc_dice = sc_dice.unsqueeze(1)
            dice_b = dice[:, sc_channel:]
            dice = torch.concat((dice_a, sc_dice, dice_b), dim=1)

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
