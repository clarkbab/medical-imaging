import torch
from typing import *

class TverskyLoss(torch.nn.Module):
    def __init__(self,
        alpha: float = 0.5,
        beta: float = 0.5,
        epsilon: float = 1e-6,
        smoothing: float = 0) -> None:
        super(TverskyLoss, self).__init__()
        self.__alpha = alpha
        self.__beta = beta
        self.__epsilon = epsilon
        self.__smoothing = smoothing

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        include_background: bool = False,
        log_fn: Optional[Callable] = None,
        mask: Optional[torch.Tensor] = None,
        reduce_channels: bool = False,
        reduction: Literal['mean', 'sum'] = 'mean',
        weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        returns: the Tversky loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of binary labels.
            mask: the B x C batch of channel masks indicating which channels are present in GT.
            weights: the B x C batch of channel weights to weight the channel-wise loss components.
        """
        if pred.shape != label.shape:
            raise ValueError(f"Expected pred shape ({pred.shape}) to equal label shape ({label.shape}).")

        # Flatten volumetric dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute smoothed label.
        label = label.type(torch.float32)
        smoothed_label = label.clone()
        smoothed_label[smoothed_label == 0] = self.__smoothing
        smoothed_label[smoothed_label == 1] = 1 - self.__smoothing

        # Compute TP, FP and FN.
        TP = (pred * smoothed_label)

        # TP loss term consists of hard TP loss, where the loss comes from voxels that would actually
        # be TPs after thresholding, and soft TP loss, where the loss comes from voxels that would be
        # FNs after thresholding.
        if log_fn is not None:
            hard_pred = pred > 0.5
            loss_hard_TP = hard_pred * TP   # Get loss TP contribution for actual TPs after thresholding.
            loss_hard_TP = loss_hard_TP.sum(dim=2)
            loss_hard_TP = loss_hard_TP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-TP-hard', loss_hard_TP.item())
            loss_soft_TP = ~hard_pred * TP   # Get loss TP contribution for actual TNs after thresholding.
            loss_soft_TP = loss_soft_TP.sum(dim=2)
            loss_soft_TP = loss_soft_TP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-TP-soft', loss_soft_TP.item())

        TP = TP.sum(dim=2)

        # Log total TP - should be sum of soft and hard losses.
        if log_fn is not None:
            loss_TP = TP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-TP', loss_TP.item())

        # Apply alpha early, so that it appears in logged values.
        FP = self.__alpha * pred * (1 - smoothed_label)
        
        # FP loss term consists of hard FP loss, where the loss comes from voxels that would actually
        # be FPs after thresholding, and soft FP loss, where the loss comes from voxels that would be
        # TNs after thresholding.
        if log_fn is not None:
            loss_hard_FP = hard_pred * FP   # Get loss FP contribution for actual FPs after thresholding.
            loss_hard_FP = loss_hard_FP.sum(dim=2)
            loss_hard_FP = loss_hard_FP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FP-hard', loss_hard_FP.item())
            loss_soft_FP = ~hard_pred * FP   # Get loss FP contribution for actual TNs after thresholding.
            loss_soft_FP = loss_soft_FP.sum(dim=2)
            loss_soft_FP = loss_soft_FP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FP-soft', loss_soft_FP.item())

        FP = FP.sum(dim=2)

        # Log total FP - should be sum of soft and hard losses.
        if log_fn is not None:
            loss_FP = FP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FP', loss_FP.item())

        FN = self.__beta * (1 - pred) * smoothed_label

        # FN loss term consists of hard FN loss, where the loss comes from voxels that would actually
        # be FNs after thresholding, and soft FN loss, where the loss comes from voxels that would be
        # TPs after thresholding.
        if log_fn is not None:
            loss_hard_FN = ~hard_pred * FN   # Get loss FN contribution for actual FNs after thresholding.
            loss_hard_FN = loss_hard_FN.sum(dim=2)
            loss_hard_FN = loss_hard_FN[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FN-hard', loss_hard_FN.item())
            loss_soft_FN = hard_pred * FN   # Get loss FN contribution for actual TNs after thresholding.
            loss_soft_FN = loss_soft_FN.sum(dim=2)
            loss_soft_FN = loss_soft_FN[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FN-soft', loss_soft_FN.item())

        FN = FN.sum(dim=2)

        # Log total FN - should be sum of soft and hard losses.
        if log_fn is not None:
            loss_FN = FN[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FN', loss_FN.item())

        # Calculate Tversky loss.
        tversky = (TP + self.__epsilon) / (TP + FP + FN + self.__epsilon)

        # Apply mask/weights across channels.
        if mask is not None:
            tversky = mask * tversky
        if weights is not None:
            tversky = weights * tversky

        # Remove background channel.
        # When all foreground classes are present, background classes can be derived from
        # these. Otherwise, background mask is empty - so it's kind of useless.
        if not include_background:
            tversky = tversky[:, 1:]

        if reduce_channels:
            # Reduce across batch and channel dimensions.
            dim = (0, 1)
        else:
            # Reduce across batch dimension only.
            dim = 0

        # Perform reduction.
        if reduction == 'mean':
            tversky = tversky.mean(dim)
        elif reduction == 'sum':    
            tversky = tversky.sum(dim)

        # Convert to loss.
        loss = 1 - tversky

        return loss

class DynamicTverskyLoss(torch.nn.Module):
    def __init__(self,
        epsilon: float = 1e-6,
        smoothing: float = 0) -> None:
        super(DynamicTverskyLoss, self).__init__()
        self.__epsilon = epsilon
        self.__smoothing = smoothing

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        params: torch.Tensor, # 2 x C.
        log_fn: Optional[Callable] = None,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        returns: the Tversky loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of binary labels.
            mask: the B x C batch of channel masks indicating which channels are present in GT.
            weights: the B x C batch of channel weights to weight the channel-wise loss components.
        """
        if pred.shape != label.shape:
            raise ValueError(f"Expected pred shape ({pred.shape}) to equal label shape ({label.shape}).")
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")

        # Flatten volumetric dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute smoothed label.
        label = label.type(torch.float32)
        smoothed_label = label.clone()
        smoothed_label[smoothed_label == 0] = self.__smoothing
        smoothed_label[smoothed_label == 1] = 1 - self.__smoothing

        # Compute TP, FP and FN.
        TP = (pred * smoothed_label)

        # TP loss term consists of hard TP loss, where the loss comes from voxels that would actually
        # be TPs after thresholding, and soft TP loss, where the loss comes from voxels that would be
        # FNs after thresholding.
        if log_fn is not None:
            hard_pred = pred > 0.5
            loss_hard_TP = hard_pred * TP   # Get loss TP contribution for actual TPs after thresholding.
            loss_hard_TP = loss_hard_TP.sum(dim=2)
            loss_hard_TP = loss_hard_TP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-TP-hard', loss_hard_TP.item())
            loss_soft_TP = ~hard_pred * TP   # Get loss TP contribution for actual TNs after thresholding.
            loss_soft_TP = loss_soft_TP.sum(dim=2)
            loss_soft_TP = loss_soft_TP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-TP-soft', loss_soft_TP.item())

        TP = TP.sum(dim=2)

        # Log total TP - should be sum of soft and hard losses.
        if log_fn is not None:
            loss_TP = TP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-TP', loss_TP.item())

        # Apply alpha early, so that it appears in logged values.
        FP = (params[0] * (pred * (1 - smoothed_label)).moveaxis(1, -1)).moveaxis(-1, 1)
        
        # FP loss term consists of hard FP loss, where the loss comes from voxels that would actually
        # be FPs after thresholding, and soft FP loss, where the loss comes from voxels that would be
        # TNs after thresholding.
        if log_fn is not None:
            loss_hard_FP = hard_pred * FP   # Get loss FP contribution for actual FPs after thresholding.
            loss_hard_FP = loss_hard_FP.sum(dim=2)
            loss_hard_FP = loss_hard_FP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FP-hard', loss_hard_FP.item())
            loss_soft_FP = ~hard_pred * FP   # Get loss FP contribution for actual TNs after thresholding.
            loss_soft_FP = loss_soft_FP.sum(dim=2)
            loss_soft_FP = loss_soft_FP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FP-soft', loss_soft_FP.item())

        FP = FP.sum(dim=2)

        # Log total FP - should be sum of soft and hard losses.
        if log_fn is not None:
            loss_FP = FP[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FP', loss_FP.item())

        FN = (params[1] * ((1 - pred) * smoothed_label).moveaxis(1, -1)).moveaxis(-1, 1)

        # FN loss term consists of hard FN loss, where the loss comes from voxels that would actually
        # be FNs after thresholding, and soft FN loss, where the loss comes from voxels that would be
        # TPs after thresholding.
        if log_fn is not None:
            loss_hard_FN = ~hard_pred * FN   # Get loss FN contribution for actual FNs after thresholding.
            loss_hard_FN = loss_hard_FN.sum(dim=2)
            loss_hard_FN = loss_hard_FN[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FN-hard', loss_hard_FN.item())
            loss_soft_FN = hard_pred * FN   # Get loss FN contribution for actual TNs after thresholding.
            loss_soft_FN = loss_soft_FN.sum(dim=2)
            loss_soft_FN = loss_soft_FN[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FN-soft', loss_soft_FN.item())

        FN = FN.sum(dim=2)

        # Log total FN - should be sum of soft and hard losses.
        if log_fn is not None:
            loss_FN = FN[:, 1:].mean()      # Remove background and take batch mean.
            log_fn('loss-FN', loss_FN.item())

        # Calculate Tversky loss.
        tversky = (TP + self.__epsilon) / (TP + FP + FN + self.__epsilon)

        # Apply mask/weights across channels.
        if mask is not None:
            tversky = mask * tversky
        if weights is not None:
            tversky = weights * tversky

        # Remove background channel.
        # When all foreground classes are present, background classes can be derived from
        # these. Otherwise, background mask is empty - so it's kind of useless.
        tversky = tversky[:, 1:]

        # Calculate loss.
        # Reduce across batch dimension only - preserve channel loss so we can see
        # how well individual channels are performing during training.
        loss = 1 - tversky.mean(0)

        # Create loss to update params.
        loss_params = (FP - FN)[:, 1:].abs().mean()

        return loss, loss_params

class TverskyDistanceLoss(torch.nn.Module):
    def __init__(self,
        alpha: float = 0.5,
        beta: float = 0.5,
        epsilon: float = 1e-6,
        smoothing: float = 0,
        dist_fn: Literal['abs', 'square'] = 'square') -> None:
        super(TverskyDistanceLoss, self).__init__()
        self.__alpha = alpha
        self.__beta = beta
        self.__epsilon = epsilon
        self.__smoothing = smoothing
        if dist_fn == 'abs':
            self.__dist_fn = torch.abs
        elif dist_fn == 'square':
            self.__dist_fn = lambda x: x ** 2
        else:
            raise ValueError(f"Unknown distance function '{dist_fn}'.")

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        returns: the Tversky loss.
        args:
            pred: the B x C x X x Y x Z batch of network predictions probabilities.
            label: the B x C x X x Y x Z batch of binary labels.
            mask: the B x C batch of channel masks indicating which channels are present in GT.
            weights: the B x C batch of channel weights to weight the channel-wise loss components.
        """
        if pred.shape != label.shape:
            raise ValueError(f"Expected pred shape ({pred.shape}) to equal label shape ({label.shape}).")
        if label.dtype != torch.bool:
            raise ValueError(f"DiceLoss expects boolean label. Got '{label.dtype}'.")

        # Flatten volumetric dimensions (X, Y, Z).
        pred = pred.flatten(start_dim=2)
        label = label.flatten(start_dim=2)

        # Compute smoothed label.
        label = label.type(torch.float32)
        smoothed_label = label.clone()
        smoothed_label[smoothed_label == 0] = self.__smoothing
        smoothed_label[smoothed_label == 1] = 1 - self.__smoothing

        # Compute TP, FP and FN.
        pred_dists = self.__dist_fn(pred - smoothed_label)
        # As 'pred_dists' can only exist in the range [0, 1 - self.__smoothing], we should scale
        # the dists so that our metrics (e.g. TP) can achieve their min/max values (0/1).
        pred_dists = pred_dists / (1 - self.__smoothing)
        # Compute 'true-positive' sum. In our analogous version, predictions become more positive
        # as the distances between predictions and smoothed label values shrink.
        # Why not use the smoothed label here? Because this would decrease our TP values, we want
        # to be able to achieve a TP of 1 when the prediction is exactly equal to the smoothed label.
        TP = ((1 - pred_dists) * label).sum(dim=2)
        FP = (pred_dists * (1 - label)).sum(dim=2)
        FN = (pred_dists * label).sum(dim=2)

        # Calculate Tversky loss.
        tversky = (TP + self.__epsilon) / (TP + self.__alpha * FP + self.__beta * FN + self.__epsilon)

        # Apply mask/weights across channels.
        if mask is not None:
            tversky = mask * tversky
        if weights is not None:
            tversky = weights * tversky

        # Remove background channel.
        # When all foreground classes are present, background classes can be derived from
        # these. Otherwise, background mask is empty - so it's kind of useless.
        tversky = tversky[:, 1:]

        # Calculate loss.
        # Reduce across batch dimension only - preserve channel loss so we can see
        # how well individual channels are performing during training.
        loss = 1 - tversky.mean(0)

        return loss
 