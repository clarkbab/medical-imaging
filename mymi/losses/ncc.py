import torch
from torch import nn

from mymi import logging

class NCCLoss(nn.Module):
    def __init__(
        self,
        epsilon: float = 1e-6,
        window: int = 9) -> None:
        super().__init__()
        logging.arg_log("Initialising NCC loss", ('epsilon', 'window'), (epsilon, window))

        self.__epsilon = epsilon
        self.__window = window

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor) -> float:
        """
        args:
            pred: the B x C x X x Y x Z batch of predicted images.
            label: the B x C x X x Y x Z batch of ground truth images.
        """
        pred_res = self.__windowed_residuals(pred)
        label_res = self.__windowed_residuals(label)
        
        # Calculate cross-correlation.
        num = (pred_res * label_res).sum(-1) ** 2
        denom = (pred_res ** 2).sum(-1) * (label_res ** 2).sum(-1)

        # Take 'mean' here (not 'sum') as each window can achieve a CC=1, and we don't
        # want varying numbers of windows to affect the loss scale.
        # 'epsilon' is required as all residuals for a window could be 0, if all image
        # intensities are the same for the window.
        ncc = ((num + self.__epsilon)/ (denom + self.__epsilon)).mean(-1)

        # Calculate mean over batch and channels.
        ncc = ncc.mean()

        # Turn into a loss!
        ncc_loss = 1 - ncc

        return ncc_loss

    def __windowed_residuals(
        self,
        t: torch.Tensor) -> torch.Tensor:
        # t: N x C x X x Y x Z
        # returns: N x C x W, window ** 3

        # Pad image with 'nans', otherwise 'unfold' will crop the image.
        spatial_dims = torch.tensor(t.shape[2:])
        rounded_dims = self.__window * torch.ceil(spatial_dims / self.__window).type(torch.int)
        pad = rounded_dims - spatial_dims
        pad = list(reversed(pad))  # 'pad' operates from back to front.
        pad = tuple(torch.tensor([[0, p] for p in pad]).flatten())
        t = torch.nn.functional.pad(t, pad, value=torch.nan)

        # Split into windows.
        n_dims = len(spatial_dims)
        for i in range(n_dims):
            t = t.unfold(i + 2, self.__window, self.__window)
        n_windows = int(torch.prod(rounded_dims / self.__window))
        n_elements = self.__window ** n_dims
        t = t.reshape(*t.shape[:2], n_windows, n_elements)     # Preserve N, C dimensions.

        # Calculate window means.
        t_mean = t.nanmean(dim=-1)

        # Calculate residuals.
        t_res = t - t_mean.unsqueeze(-1)

        # Replace 'nan' residuals with 0. These won't contribute to the loss.
        t_res = t_res.nan_to_num(0)

        return t_res
