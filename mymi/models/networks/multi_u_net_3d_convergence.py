from fairscale.nn import checkpoint_wrapper
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from typing import Optional

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_res):
        x = self.upsample(x)

        # Spatial resolution may be lost due to rounding when downsampling. Pad the upsampled features
        # if necessary.
        if x.shape != x_res.shape:
            n_axes = len(x.shape)
            padding = np.zeros((n_axes, 2), dtype='uint8')
            for axis in range(n_axes):
                diff = x_res.shape[axis] - x.shape[axis]
                if diff > 0:
                    padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
            padding = tuple(np.flip(padding, axis=0).flatten())
            x = pad(x, padding)

        x = torch.cat((x, x_res), dim=1)
        x = self.double_conv(x)

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class MultiUNet3DConvergence(nn.Module):
    def __init__(
        self,
        n_output_channels: int,
        **kwargs):
        super().__init__()

        # Define layers.
        n_features = 64
        self.first = DoubleConv(1, n_features)
        self.down1 = Down(n_features, 2 * n_features)
        self.down2 = Down(2 * n_features, 4 * n_features)
        self.down3 = Down(4 * n_features, 8 * n_features)
        self.down4 = Down(8 * n_features, 16 * n_features)
        self.up1 = Up(16 * n_features, 8 * n_features)
        self.up2 = Up(8 * n_features, 4 * n_features)
        self.up3 = Up(4 * n_features, 2 * n_features)
        self.up4 = Up(2 * n_features, n_features)
        self.out = OutConv(n_features, n_output_channels)
        self.softmax = nn.Softmax(dim=1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        x = self.softmax(x)
        return x
