import torch
from typing import List

def add_background(labels: torch.Tensor, foreground_channels: List[int]) -> torch.Tensor:
    """
    Prepends a background channel to a label tensor.

    args:
        labels: B x C x <spatial> tensor of label maps.
        foreground_channels: channel indices in `labels` that constitute foreground.
    returns: B x (1 + len(foreground_channels)) x <spatial> tensor, where channel 0
        is the background (1 - union of foreground) and remaining channels are the
        specified foreground channels in order.
    """
    foreground = labels[:, foreground_channels]
    background = 1 - foreground.amax(dim=1, keepdim=True)
    return torch.cat([background, foreground], dim=1)
