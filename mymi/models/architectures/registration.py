import torch
from typing import *

from .unet3d import UNet3D

class Resampler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x_moving: torch.Tensor,
        y_dvf: torch.Tensor) -> torch.Tensor:
        # x_moving: N, 1, X, Y, Z.
        # y_dvf: N, 3, X, Y, Z.

        # Create identity grid.
        spatial_shape = y_dvf.shape[2:]
        identity = torch.meshgrid([torch.linspace(-1, 1, s) for s in spatial_shape], indexing='ij')
        identity = torch.stack(identity)
        identity = torch.stack([identity] * len(x_moving))  # Duplicate for each batch item.
        identity = identity.to(y_dvf.device)

        # Create total grid.
        grid = identity + y_dvf
        grid = grid.moveaxis(1, -1)     # Channels at the back.

        # 'grid_sample' expects grid spatial element (x, y, z) to hold deformation vector (dz, dy, dx).
        grid = grid.flip(-1)

        # Subtract/add true padding value as 'grid_sample' will always zero-pad.
        # pad = x_moving.min() if self.__pad_value == 'min' else self.__pad_value
        # x_moving = x_moving - pad
        # Why 'padding_mode=border'?
        # We want to discourage DVF to point offscreen of moving image. Initially we achieved this by padding with -2000,
        # which doesn't match with anything to discourage offscreen vectors. Border padding should also achieve this, as 
        # an offscreen intensity that can be reached by a vector (to maximise voxel similarity) can always be achieved by
        # a closer onscreen voxel - coupled with DVF smoothing, this should preference onscreen voxels.
        y_moved = torch.nn.functional.grid_sample(x_moving, grid, align_corners=True, padding_mode='border')
        # y_moved = y_moved + pad

        return y_moved

class RegMod(torch.nn.Module):
    def __init__(
        self,
        **kwargs) -> None:
        super().__init__()
        n_input_channels = 2
        n_output_channels = 3
        # 'use_small_output_params' reduces the scale of the DVF prediction initially. This ensures that our transformation
        # is approximately an identity transform and gives the network a reasonable starting point.
        self.__network = UNet3D(n_output_channels, n_input_channels=n_input_channels, use_softmax=False, use_small_output_params=True, **kwargs)
        self.__resampler = Resampler()

    def forward(
        self,
        input: torch.Tensor) -> torch.Tensor:
        y_dvf = self.__network(input)
        x_moving = input[:, 1].unsqueeze(1)
        y_moved = self.__resampler(x_moving, y_dvf)

        return y_moved, y_dvf
