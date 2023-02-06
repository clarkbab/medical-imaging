from asyncio.trsock import TransportSocket
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from torch.nn.functional import pad
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from fairscale.nn.checkpoint import checkpoint_wrapper
from typing import Callable, List

from mymi.utils import gpu_usage

class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        super().__init__()

        self.double_conv = Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels)
        )

    def forward(self, x, dummy_arg=None):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        super().__init__()

        self.down = Sequential(
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

        self.upsample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(2 * out_channels, out_channels)

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

class PrintWrapper(nn.Module):
    def __init__(self, module, name):
        super().__init__()
        self.__module = module
        self.__name = name
        self.__gpu_available = torch.cuda.is_available()

    def forward(self, *params):
        print(f"=== layer: {self.__name} ===")
        for i, param in enumerate(params):
            print(f"input {i} shape/dtype: {param.shape}/{param.dtype}")
        if self.__gpu_available:
            gpu_before = gpu_usage()
        y = self.__module(*params)
        print(f"output shape/dtype: {y.shape}/{y.dtype}")
        if self.__gpu_available:
            gpu_after = gpu_usage()
            gpu_diff = [ga - gb for ga, gb in zip(gpu_after, gpu_before)]
            print(f"gpu diff/total (MB): {gpu_diff[0]:.2f}/{gpu_after[0]:.2f}")
        return y

class Encoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.first = PrintWrapper(DoubleConv(1, n_features), 'first')
        self.down1 = PrintWrapper(Down(n_features, 2 * n_features), 'down1')
        self.down2 = PrintWrapper(Down(2 * n_features, 4 * n_features), 'down2')
        self.down3 = PrintWrapper(Down(4 * n_features, 8 * n_features), 'down3')
        self.down4 = PrintWrapper(Down(8 * n_features, 16 * n_features), 'down4')

    def forward(self, x):
        x0 = self.first(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x0, x1, x2, x3, x4

class Decoder(nn.Module):
    def __init__(self, n_features, n_output_channels):
        super().__init__()
        self.up1 = PrintWrapper(Up(16 * n_features, 8 * n_features), 'up1')
        self.up2 = PrintWrapper(Up(8 * n_features, 4 * n_features), 'up2')
        self.up3 = PrintWrapper(Up(4 * n_features, 2 * n_features), 'up3')
        self.up4 = PrintWrapper(Up(2 * n_features, n_features), 'up4')
        self.out = PrintWrapper(OutConv(n_features, n_output_channels), 'out')
        self.softmax = PrintWrapper(nn.Softmax(dim=1), 'softmax')

    def forward(self, x0, x1, x2, x3, x4):
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.out(x)
        return self.softmax(x)

class MutilUNet3DMemoryTest(nn.Module):
    def __init__(
        self,
        n_output_channels: int) -> None:
        super().__init__()

        # Define layers.
        n_features = 32
        self.encoder = Encoder(n_features)
        self.decoder = Decoder(n_features, n_output_channels)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        return self.decoder(x0, x1, x2, x3, x4)
