from asyncio.trsock import TransportSocket
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from typing import Optional

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

        self.double_conv = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels)
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

class MultiUNet3D(nn.Module):
    def __init__(
        self,
        n_output_channels: int,
        n_gpus: int = 1) -> None:
        super().__init__()
        self.__n_gpus = n_gpus

        # Define layers.
        n_features = 32

        # Split 'first' to place on separate GPUs.
        self.first_a = nn.Conv3d(in_channels=1, out_channels=n_features, kernel_size=3, stride=1, padding=1)
        self.first_b = nn.InstanceNorm3d(n_features)
        self.first_c = nn.ReLU()
        self.first_d = nn.Conv3d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1)
        self.first_e = nn.InstanceNorm3d(n_features)
        self.first_f = nn.ReLU()

        # Split 'down1' to place on separate GPUs.
        self.down1_a = nn.MaxPool3d(kernel_size=2)
        self.down1_b = Conv(n_features, 2 * n_features)
        self.down1_c = Conv(2 * n_features, 2 * n_features)

        self.down2 = Down(2 * n_features, 4 * n_features)
        self.down3 = Down(4 * n_features, 8 * n_features)
        self.down4 = Down(8 * n_features, 16 * n_features)
        self.up1 = Up(16 * n_features, 8 * n_features)
        self.up2 = Up(8 * n_features, 4 * n_features)

        # Split 'up3' to place on separate GPUs.
        self.up3_a = nn.ConvTranspose3d(in_channels=4 * n_features, out_channels=2 * n_features, kernel_size=2, stride=2) 
        self.up3_b = Conv(4 * n_features, 2 * n_features)
        self.up3_c = Conv(2 * n_features, 2 * n_features)

        # Split 'up4' to place on separate GPUs.
        self.up4_a = nn.ConvTranspose3d(in_channels=2 * n_features, out_channels=n_features, kernel_size=2, stride=2) 
        self.up4_b = nn.Conv3d(in_channels=2 * n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1)
        self.up4_c = nn.InstanceNorm3d(n_features)
        self.up4_d = nn.ReLU()
        self.up4_e = nn.Conv3d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1)
        self.up4_f = nn.InstanceNorm3d(n_features)
        self.up4_g = nn.ReLU()

        self.out = OutConv(n_features, n_output_channels)
        self.softmax = nn.Softmax(dim=1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Assign devices based on number of GPUs.
        if self.__n_gpus == 1:
            device_0, device_1, device_2, device_3 = 4 * ['cuda:0']
        elif self.__n_gpus == 2:
            device_0, device_2, device_1, device_3 = 2 * ['cuda:0', 'cuda:1']
        elif self.__n_gpus == 4:
            device_0, device_2, device_1, device_3 = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

        # Split 'first' layer to place on separate GPUs.
        x = self.first_a(x).to(device_0)
        x = self.first_b(x).to(device_0)
        x = self.first_c(x).to(device_0)
        x = self.first_d(x).to(device_0)
        x = self.first_e(x).to(device_0)
        x1 = self.first_f(x).to(device_1)

        # Split 'down1' layer to place on separate GPUs.
        x = self.down1_a(x1).to(device_1)
        x = self.down1_b(x).to(device_1)
        x2 = self.down1_c(x).to(device_1)

        x3 = self.down2(x2).to(device_1)
        x4 = self.down3(x3).to(device_1)
        x = self.down4(x4).to(device_1)
        x = self.up1(x, x4).to(device_1)
        x = self.up2(x, x3).to(device_1)

        # Split 'up3' layer to place on separate GPUs.
        x = self.up3_a(x).to(device_1)
        if x.shape != x2.shape:
            n_axes = len(x.shape)
            padding = np.zeros((n_axes, 2), dtype='uint8')
            for axis in range(n_axes):
                diff = x2.shape[axis] - x.shape[axis]
                if diff > 0:
                    padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
            padding = tuple(np.flip(padding, axis=0).flatten())
            x = pad(x, padding)
        x = torch.cat((x, x2), dim=1).to(device_1)
        x = self.up3_b(x).to(device_1)
        x = self.up3_c(x).to(device_2)

        # Split 'up4' layer to place on separate GPUs.
        x = self.up4_a(x).to(device_2)
        if x.shape != x1.shape:
            n_axes = len(x.shape)
            padding = np.zeros((n_axes, 2), dtype='uint8')
            for axis in range(n_axes):
                diff = x1.shape[axis] - x.shape[axis]
                if diff > 0:
                    padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
            padding = tuple(np.flip(padding, axis=0).flatten())
            x = pad(x, padding)
        x = torch.cat((x, x1), dim=1).to(device_2)
        x = self.up4_b(x).to(device_2)
        x = self.up4_c(x).to(device_2)
        x = self.up4_d(x).to(device_2)
        x = self.up4_e(x).to(device_3)
        x = self.up4_f(x).to(device_3)
        x = self.up4_g(x).to(device_3)

        x = self.out(x).to(device_3)
        x = self.softmax(x).to(device_3)
        return x
