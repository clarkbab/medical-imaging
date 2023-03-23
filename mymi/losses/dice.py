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
        batch_size = pred.shape[0]
        if weights is not None:
            assert weights.shape == pred.shape[:2]
            for b in range(batch_size):
                weight_sum = weights[b].sum().round(decimals=3)
                if weight_sum != 1:
                    raise ValueError(f"Weights (batch={b}) must sum to 1. Got '{weight_sum}' (weights={weights[b]}).")
        assert reduction in ('mean', 'sum')

        # Separate 'SpinalCord' and 'background' channels if necessary.
        # If all regions are present, the 'background' will be the inverse of the union of all regions.
        # To encourage a long 'SpinalCord' prediction, we have to discount any loss due to voxels below
        # the ground truth for 'SpinalCord' for both 'SpinalCord' and 'background' classes.
        if sc_channel is not None:
            assert sc_channel >= 1
            # Split 'SpinalCord' and 'background' channels out from main data.
            bk_pred = pred[:, 0]
            sc_pred = pred[:, sc_channel]
            bk_label = label[:, 0]
            sc_label = label[:, sc_channel]
            pred_a = pred[:, 1:sc_channel]
            label_a = label[:, 1:sc_channel]
            pred_b = pred[:, (sc_channel + 1):]
            label_b = label[:, (sc_channel + 1):]
            pred = torch.concat((pred_a, pred_b), dim=1)
            label = torch.concat((label_a, label_b), dim=1)

            # Separate 'SpinalCord' and 'background' labels along batch dimension as each will have different 'label_min_z'.
            bk_preds = []
            sc_preds = []
            bk_labels = []
            sc_labels = []
            for i in range(batch_size):
                # Remove all voxels below 'label_min_z' from loss calculation.
                # We don't want to penalise the model for predicting further in inferior
                # direction than the label.
                label_min_z = torch.argwhere(sc_label[i]).min(axis=0).values[2].item()
                bk_pred_i = bk_pred[i, :, :, label_min_z:]
                sc_pred_i = sc_pred[i, :, :, label_min_z:]
                bk_label_i = bk_label[i, :, :, label_min_z:]
                sc_label_i = sc_label[i, :, :, label_min_z:]
                bk_preds.append(bk_pred_i)
                sc_preds.append(sc_pred_i)
                bk_labels.append(bk_label_i)
                sc_labels.append(sc_label_i)

        # Flatten spatial dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)
        if sc_channel is not None:
            for i in range(batch_size):
                bk_preds[i] = bk_preds[i].flatten()
                sc_preds[i] = sc_preds[i].flatten()
                bk_labels[i] = bk_labels[i].flatten()
                sc_labels[i] = sc_labels[i].flatten()

        # Compute dice loss.
        intersection = (pred * label).sum(dim=2)
        denominator = (pred + label).sum(dim=2)
        loss = -(2. * intersection + self.__epsilon) / (denominator + self.__epsilon)
        if sc_channel is not None:
            sc_losses = []
            bk_losses = []
            for i in range(batch_size):
                # Calculate 'SpinalCord' loss.
                bk_intersection = (bk_preds[i] * bk_labels[i]).sum()
                sc_intersection = (sc_preds[i] * sc_labels[i]).sum()
                bk_denominator = (bk_preds[i] + bk_labels[i]).sum()
                sc_denominator = (sc_preds[i] + sc_labels[i]).sum()
                bk_loss = -(2. * bk_intersection + self.__epsilon) / (bk_denominator + self.__epsilon)
                sc_loss = -(2. * sc_intersection + self.__epsilon) / (sc_denominator + self.__epsilon)
                bk_losses.append(bk_loss)
                sc_losses.append(sc_loss)

            # Insert result back into main dice results.
            loss_a = loss[:, :sc_channel]
            bk_loss = torch.Tensor(bk_losses).unsqueeze(1)
            sc_loss = torch.Tensor(sc_losses).unsqueeze(1)
            loss_b = loss[:, sc_channel:]
            loss = torch.concat((bk_loss, loss_a, sc_loss, loss_b), dim=1)

        # Apply mask across channels.
        if mask is not None:
            loss = mask * loss

        # Apply weights across channels.
        if weights is not None:
            loss = weights * loss

        # Reduce across batch and channel dimensions.
        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        return loss
