from typing import Optional

import torch
import torch.nn.functional as F

from mymi import logging

class HeatmapKLDivLoss(torch.nn.Module):
    """KL divergence between a per-channel spatial softmax of raw model logits and
    normalised target heatmaps (each channel treated as a discrete 2D distribution)."""
    def __init__(self) -> None:
        super().__init__()
        logging.info("Initialising Heatmap KL-divergence loss.")

    def forward(
        self,
        pred: torch.Tensor,     # B x C x H x W - raw logits.
        label: torch.Tensor,    # B x C x H x W - target probabilities, summing to 1 per channel.
        mask: Optional[torch.Tensor] = None,   # B x C - True where the channel's marker is valid and should contribute to the loss.
        ) -> torch.Tensor:
        assert pred.shape == label.shape, f"Expected 'pred' and 'label' to have the same shape, got {pred.shape} and {label.shape}."
        b, c, h, w = pred.shape
        log_pred = F.log_softmax(pred.reshape(b, c, h * w), dim=-1)
        target = label.reshape(b, c, h * w)
        if mask is None:
            return F.kl_div(log_pred, target, reduction='batchmean')

        # Per-channel KL, masked to exclude invalid (e.g. off-grid) markers before averaging.
        kl = F.kl_div(log_pred, target, reduction='none').sum(dim=-1)   # B x C.
        n_valid = mask.sum()
        if n_valid == 0:
            return kl.sum() * 0.0   # No valid markers in batch - zero loss, preserves grad graph.
        return (kl * mask).sum() / n_valid
