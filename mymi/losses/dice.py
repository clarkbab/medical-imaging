import torch
from torch import nn
from torch.nn import functional as F
from typing import *

from mymi import logging

class DiceLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-6,
        smoothing: float = 0) -> None:
        logging.info(f"Initialising DiceLoss with epsilon={epsilon}, smoothing={smoothing}.")
        super().__init__()
        self.__epsilon = epsilon
        self.__smoothing = smoothing

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        include_background: bool = False,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        reduce_channels: bool = False,
        reduction: Literal['mean', 'sum'] = 'mean',
        ) -> torch.Tensor:
        """
        returns: the dice loss.
        args:
            pred: the B x C x <spatial> batch of network prediction probabilities (2D or 3D).
            label: the B x C x <spatial> batch of one-hot-encoded labels (2D or 3D).
        """
        if pred.dim() not in (4, 5):
            raise ValueError(f"DiceLoss expects 4D (2D) or 5D (3D) pred tensor, got shape {tuple(pred.shape)}.")
        if label.shape != pred.shape:
            if label.shape != (pred.shape[0], *pred.shape[2:]):
                raise ValueError(f"DiceLoss expects label shape {tuple(pred.shape)} (one-hot) or {(pred.shape[0], *pred.shape[2:])} (categorical), got {tuple(label.shape)}.")

            # One-hot encode the label.
            label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
            label = F.one_hot(label, num_classes=2)
            label = label.movedim(-1, 1)
        if mask is not None:
            assert mask.shape == label.shape[:2]
        batch_size = label.shape[0]
        if weights is not None:
            assert weights.shape == label.shape[:2]
            for b in range(batch_size):
                weight_sum = weights[b].sum().round(decimals=3)
                if weight_sum != 1:
                    raise ValueError(f"Weights (batch={b}) must sum to 1. Got '{weight_sum}' (weights={weights[b]}).")
        assert reduction in ('mean', 'sum')

        # Flatten spatial dimensions.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Smooth the label.
        label = label.type(torch.float32)
        smoothed_label = label.clone()
        smoothed_label[smoothed_label == 0] = self.__smoothing
        smoothed_label[smoothed_label == 1] = 1 - self.__smoothing

        # Compute dice loss.
        intersection = (pred * smoothed_label).sum(dim=2)
        denominator = (pred + smoothed_label).sum(dim=2)
        loss = 1 - (2. * intersection + self.__epsilon) / (denominator + self.__epsilon)

        # Apply mask across channels.
        if mask is not None:
            loss = mask * loss

        # Apply weights across channels.
        if weights is not None:
            loss = weights * loss

        # Remove background channel before computing final loss.
        if not include_background:
            loss = loss[:, 1:]

        if reduce_channels:
            # Reduce across batch and channel dimensions.
            dim = (0, 1)
        else:
            # Reduce across batch dimension only.
            dim = 0

        if reduction == 'mean':
            loss = loss.mean(dim)
        elif reduction == 'sum':
            loss = loss.sum(dim)

        return loss
    
class DML1Loss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-6,
        smoothing: float = 0.1) -> None:
        logging.info(f"Initialising DML1Loss with epsilon={epsilon}, smoothing={smoothing}.")
        super().__init__()
        self.__epsilon = epsilon
        self.__smoothing = smoothing

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        reduce_channels: bool = False,
        reduction: Literal['mean', 'sum'] = 'mean') -> float:
        """
        returns: the dice loss.
        args:
            pred: the B x C x <spatial> batch of network prediction probabilities (2D or 3D).
            label: the B x C x <spatial> batch of one-hot-encoded labels (2D or 3D).
        """
        if pred.dim() not in (4, 5):
            raise ValueError(f"DML1Loss expects 4D (2D) or 5D (3D) pred tensor, got shape {tuple(pred.shape)}.")
        if label.dtype != torch.bool:
            raise ValueError(f"DML1Loss expects boolean label. Got '{label.dtype}'.")
        if pred.dtype == torch.bool:
            raise ValueError(f"Pred should have float type, not bool.")
        if label.shape != pred.shape:
            if label.shape != (pred.shape[0], *pred.shape[2:]):
                raise ValueError(f"DML1Loss expects label shape {tuple(pred.shape)} (one-hot) or {(pred.shape[0], *pred.shape[2:])} (categorical), got {tuple(label.shape)}.")

            # One-hot encode the label.
            label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
            label = F.one_hot(label, num_classes=2)
            label = label.movedim(-1, 1)
        if mask is not None:
            assert mask.shape == label.shape[:2]
        batch_size = label.shape[0]
        if weights is not None:
            assert weights.shape == label.shape[:2]
            for b in range(batch_size):
                weight_sum = weights[b].sum().round(decimals=3)
                if weight_sum != 1:
                    raise ValueError(f"Weights (batch={b}) must sum to 1. Got '{weight_sum}' (weights={weights[b]}).")
        assert reduction in ('mean', 'sum')

        # Flatten spatial dimensions.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute smoothed label.
        label = label.type(torch.float32)
        smoothed_label = label.clone()
        smoothed_label[smoothed_label == 0] = self.__smoothing
        smoothed_label[smoothed_label == 1] = 1 - self.__smoothing

        # Compute DML1.
        # Don't need abs for L1 norms as pred and smoothed label are positive.
        diffs = torch.abs(pred - smoothed_label)
        diffs = (pred - smoothed_label) ** 2
        numer = torch.sum(pred, dim=2) + torch.sum(smoothed_label, dim=2) + torch.sum(torch.abs(pred - smoothed_label), dim=2)
        denom = torch.sum(pred, dim=2) + torch.sum(smoothed_label, dim=2)
        dice = (numer + self.__epsilon) / (denom + self.__epsilon)
        loss = 1 - dice

        # Apply mask across channels.
        if mask is not None:
            loss = mask * loss

        # Apply weights across channels.
        if weights is not None:
            loss = weights * loss

        if reduce_channels:
            # Reduce across batch and channel dimensions.
            dim = (0, 1)
        else:
            # Reduce across batch dimension only.
            dim = 0

        # If multi-channel, remove the background channel as this
        # can be computed from the other channels.
        if loss.shape[1] > 1:
            loss = loss[:, 1:]

        if reduction == 'mean':
            loss = loss.mean(dim)
        elif reduction == 'sum':
            loss = loss.sum(dim)

        return loss

class DML2Loss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-6,
        smoothing: float = 0.1) -> None:
        logging.info(f"Initialising DML2Loss with epsilon={epsilon}, smoothing={smoothing}.")
        super().__init__()
        self.__epsilon = epsilon
        self.__smoothing = smoothing

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
        reduce_channels: bool = False,
        reduction: Literal['mean', 'sum'] = 'mean') -> float:
        """
        returns: the dice loss.
        args:
            pred: the B x C x <spatial> batch of network prediction probabilities (2D or 3D).
            label: the B x C x <spatial> batch of one-hot-encoded labels (2D or 3D).
        """
        if pred.dim() not in (4, 5):
            raise ValueError(f"DML2Loss expects 4D (2D) or 5D (3D) pred tensor, got shape {tuple(pred.shape)}.")
        if label.dtype != torch.bool:
            raise ValueError(f"DML2Loss expects boolean label. Got '{label.dtype}'.")
        if pred.dtype == torch.bool:
            raise ValueError(f"Pred should have float type, not bool.")
        if label.shape != pred.shape:
            if label.shape != (pred.shape[0], *pred.shape[2:]):
                raise ValueError(f"DML2Loss expects label shape {tuple(pred.shape)} (one-hot) or {(pred.shape[0], *pred.shape[2:])} (categorical), got {tuple(label.shape)}.")

            # One-hot encode the label.
            label = label.long()    # 'F.one_hot' Expects dtype 'int64'.
            label = F.one_hot(label, num_classes=2)
            label = label.movedim(-1, 1)
        if mask is not None:
            assert mask.shape == label.shape[:2]
        batch_size = label.shape[0]
        if weights is not None:
            assert weights.shape == label.shape[:2]
            for b in range(batch_size):
                weight_sum = weights[b].sum().round(decimals=3)
                if weight_sum != 1:
                    raise ValueError(f"Weights (batch={b}) must sum to 1. Got '{weight_sum}' (weights={weights[b]}).")
        assert reduction in ('mean', 'sum')

        # Flatten spatial dimensions.
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute smoothed label.
        label = label.type(torch.float32)
        smoothed_label = label.clone()
        smoothed_label[smoothed_label == 0] = self.__smoothing
        smoothed_label[smoothed_label == 1] = 1 - self.__smoothing

        # Compute DML2.
        # Don't need abs for L1 norms as pred and smoothed label are positive.
        # diffs = torch.abs(pred - smoothed_label)
        diffs = (pred - smoothed_label) ** 2
        numer = 2 * torch.sum(pred * smoothed_label, dim=2)
        denom = 2 * torch.sum(pred * smoothed_label, dim=2) + torch.sum(diffs, dim=2)
        dice = (numer + self.__epsilon) / (denom + self.__epsilon)
        loss = 1 - dice

        # Apply mask across channels.
        if mask is not None:
            loss = mask * loss

        # Apply weights across channels.
        if weights is not None:
            loss = weights * loss

        if reduce_channels:
            # Reduce across batch and channel dimensions.
            dim = (0, 1)
        else:
            # Reduce across batch dimension only.
            dim = 0

        # If multi-channel, remove the background channel as this
        # can be computed from the other channels.
        if loss.shape[1] > 1:
            loss = loss[:, 1:]

        if reduction == 'mean':
            loss = loss.mean(dim)
        elif reduction == 'sum':
            loss = loss.sum(dim)

        return loss
