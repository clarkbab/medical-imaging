from asyncio.trsock import TransportSocket
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from torch.nn.functional import pad
from torch import Tensor
from torch.utils.checkpoint import checkpoint
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

    def forward(self, x):
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

    def forward(self, *params):
        print(f"=== layer: {self.__name} ===")
        for i, param in enumerate(params):
            print(f"input {i} shape/dtype: {param.shape}/{param.dtype}")
        gpu_before = gpu_usage()
        y = self.__module(*params)
        print(f"output shape/dtype: {y.shape}/{y.dtype}")
        gpu_after = gpu_usage()
        gpu_diff = [ga - gb for ga, gb in zip(gpu_after, gpu_before)]
        # a.element_size() * a.nelement().
        print(f"gpu diff/total (MB): {gpu_diff[0]:.2f}/{gpu_after[0]:.2f}")
        return y

class MultiUNet3D(nn.Module):
    def __init__(
        self,
        n_output_channels: int,
        n_gpus: int = 0) -> None:
        super().__init__()
        self.__n_gpus = n_gpus

        # Assign devices based on number of GPUs.
        if self.__n_gpus == 0:
            self.__device_0, self.__device_1, self.__device_2, self.__device_3 = 4 * ['cpu']
        if self.__n_gpus == 1:
            self.__device_0, self.__device_1, self.__device_2, self.__device_3 = 4 * ['cuda:0']
        elif self.__n_gpus == 2:
            self.__device_0, self.__device_2, self.__device_1, self.__device_3 = 2 * ['cuda:0', 'cuda:1']
        elif self.__n_gpus == 4:
            self.__device_0, self.__device_1, self.__device_2, self.__device_3 = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']

        # Define layers.
        n_features = 32

        # Split 'first' to place on separate GPUs.
        self.first_a = PrintWrapper(nn.Conv3d(in_channels=1, out_channels=n_features, kernel_size=3, stride=1, padding=1).to(self.__device_0), 'first_a')
        self.first_b = PrintWrapper(nn.InstanceNorm3d(n_features), 'first_b')
        self.first_c = PrintWrapper(nn.ReLU(), 'first_c')
        self.first_d = PrintWrapper(nn.Conv3d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1).to(self.__device_0), 'first_d')
        self.first_e = PrintWrapper(nn.InstanceNorm3d(n_features), 'first_e')
        self.first_f = PrintWrapper(nn.ReLU(), 'first_f')

        # Split 'down1' to place on separate GPUs.
        self.down1_a = PrintWrapper(nn.MaxPool3d(kernel_size=2), 'down1_a')
        self.down1_b = PrintWrapper(Conv(n_features, 2 * n_features).to(self.__device_1), 'down1_b')
        self.down1_c = PrintWrapper(Conv(2 * n_features, 2 * n_features).to(self.__device_1), 'down1_c')

        self.down2 = PrintWrapper(Down(2 * n_features, 4 * n_features).to(self.__device_1), 'down2')
        self.down3 = PrintWrapper(Down(4 * n_features, 8 * n_features).to(self.__device_1), 'down3')
        self.down4 = PrintWrapper(Down(8 * n_features, 16 * n_features).to(self.__device_1), 'down4')
        self.up1 = PrintWrapper(Up(16 * n_features, 8 * n_features).to(self.__device_1), 'up1')
        self.up2 = PrintWrapper(Up(8 * n_features, 4 * n_features).to(self.__device_1), 'up2')

        # Split 'up3' to place on separate GPUs.
        self.up3_a = PrintWrapper(nn.ConvTranspose3d(in_channels=4 * n_features, out_channels=2 * n_features, kernel_size=2, stride=2).to(self.__device_1), 'up3_a')
        self.up3_b = PrintWrapper(Conv(4 * n_features, 2 * n_features).to(self.__device_1), 'up3_b')
        self.up3_c = PrintWrapper(Conv(2 * n_features, 2 * n_features).to(self.__device_1), 'up3_c')

        # Split 'up4' to place on separate GPUs.
        self.up4_a = PrintWrapper(nn.ConvTranspose3d(in_channels=2 * n_features, out_channels=n_features, kernel_size=2, stride=2).to(self.__device_1), 'up4_a')
        self.up4_b = PrintWrapper(nn.Conv3d(in_channels=2 * n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1).to(self.__device_2), 'up4_b')
        self.up4_c = PrintWrapper(nn.InstanceNorm3d(n_features), 'up4_c')
        self.up4_d = PrintWrapper(nn.ReLU(), 'up4_d')
        self.up4_e = PrintWrapper(nn.Conv3d(in_channels=n_features, out_channels=n_features, kernel_size=3, stride=1, padding=1).to(self.__device_3), 'up4_e')
        self.up4_f = PrintWrapper(nn.InstanceNorm3d(n_features), 'up4_f')
        self.up4_g = PrintWrapper(nn.ReLU(), 'up4_g')

        self.out = PrintWrapper(OutConv(n_features, n_output_channels).to(self.__device_3), 'out')
        self.softmax = PrintWrapper(nn.Softmax(dim=1), 'softmax')

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_ckpt_func(
        self,
        layers: List[Module]) -> Callable:
        model = Sequential(*layers)
        def ckpt_func(x, dummy_arg) -> Tensor:
            assert dummy_arg is not None
            return model(x)
        return ckpt_func

    def forward(self, x, dummy_arg=None):
        # assert dummy_arg is not None
        # # Split 'first' layer to place on separate GPUs.
        # x = self.first_a(x.to(self.__device_0))
        # x = self.first_b(x)
        # x = self.first_c(x)
        # x = self.first_d(x)
        # x = self.first_e(x)
        # x1 = self.first_f(x)
        dummy_arg = torch.Tensor()
        dummy_arg.requires_grad = True
        ckpt1_layers = [
            self.first_a,
            self.first_b,
            self.first_c
        ]
        x = checkpoint(self.get_ckpt_func(ckpt1_layers), x.to(self.__device_0), dummy_arg)
        ckpt2_layers = [
            self.first_d,
            self.first_e,
            self.first_f
        ]
        x1 = checkpoint(self.get_ckpt_func(ckpt2_layers), x)

        # # Split 'down1' layer to place on separate GPUs.
        # x = self.down1_a(x1.to(self.__device_1))
        # x = self.down1_b(x)
        # x2 = self.down1_c(x)

        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x = self.down4(x4)

        def ckpt3_func(xi):
            x = self.down1_a(xi.to(self.__device_1))
            x = self.down1_b(x)
            x2 = self.down1_c(x)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.down4(x4)
            return x, x2, x3, x4
        x, x2, x3, x4 = checkpoint(ckpt3_func, x1)

        # x = self.up1(x, x4)
        # x = self.up2(x, x3)

        # # Split 'up3' layer to place on separate GPUs.
        # x = self.up3_a(x)
        # if x.shape != x2.shape:
        #     n_axes = len(x.shape)
        #     padding = np.zeros((n_axes, 2), dtype='uint8')
        #     for axis in range(n_axes):
        #         diff = x2.shape[axis] - x.shape[axis]
        #         if diff > 0:
        #             padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
        #     padding = tuple(np.flip(padding, axis=0).flatten())
        #     x = pad(x, padding)
        # x = torch.cat((x, x2), dim=1)
        # x = self.up3_b(x)
        # x = self.up3_c(x)
        
        def ckpt4_func(xi, xi2, xi3, xi4):
            x = self.up1(xi, xi4)
            x = self.up2(x, xi3)

            # Split 'up3' layer to place on separate GPUs.
            x = self.up3_a(x)
            if x.shape != xi2.shape:
                n_axes = len(x.shape)
                padding = np.zeros((n_axes, 2), dtype='uint8')
                for axis in range(n_axes):
                    diff = xi2.shape[axis] - x.shape[axis]
                    if diff > 0:
                        padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
                padding = tuple(np.flip(padding, axis=0).flatten())
                x = pad(x, padding)
            x = torch.cat((x, xi2), dim=1)
            x = self.up3_b(x)
            x = self.up3_c(x)
            return x
        x = checkpoint(ckpt4_func, x, x2, x3, x4)

        # # Split 'up4' layer to place on separate GPUs.
        # x = self.up4_a(x)
        # if x.shape != x1.shape:
        #     n_axes = len(x.shape)
        #     padding = np.zeros((n_axes, 2), dtype='uint8')
        #     for axis in range(n_axes):
        #         diff = x1.shape[axis] - x.shape[axis]
        #         if diff > 0:
        #             padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
        #     padding = tuple(np.flip(padding, axis=0).flatten())
        #     x = pad(x, padding)
        # x = torch.cat((x, x1.to(self.__device_1)), dim=1)

        # x = self.up4_b(x.to(self.__device_2))
        # x = self.up4_c(x)
        # x = self.up4_d(x)
        # x = self.up4_e(x.to(self.__device_3))
        # x = self.up4_f(x)
        # x = self.up4_g(x)

        # Split 'up4' layer to place on separate GPUs.
        def ckpt5_func(xi, xi1):
            x = self.up4_a(xi)
            if x.shape != xi1.shape:
                n_axes = len(x.shape)
                padding = np.zeros((n_axes, 2), dtype='uint8')
                for axis in range(n_axes):
                    diff = xi1.shape[axis] - x.shape[axis]
                    if diff > 0:
                        padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
                padding = tuple(np.flip(padding, axis=0).flatten())
                x = pad(x, padding)
            x = torch.cat((x, xi1.to(self.__device_1)), dim=1)
            x = self.up4_b(x.to(self.__device_2))
            x = self.up4_c(x)
            return x
        x = checkpoint(ckpt5_func, x, x1)

        ckpt6_layers = [
            self.up4_d,
            self.up4_e,
            self.up4_f,
            self.up4_g
        ]
        x = checkpoint(self.get_ckpt_func(ckpt6_layers), x)

        x = self.out(x)
        x = self.softmax(x)

        return x
