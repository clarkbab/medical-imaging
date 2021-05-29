import logging
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: bool = False):
        """
        args:
            in_channels: the number of channels into the module.
            out_channels: the number of channels out of the module.
        kwargs:
            dropout: include dropout layer.
        """
        super().__init__()

        # Define convolution blocks.
        convs = [
            nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            )
        ]

        # Add dropout if necessary.
        if dropout:
            for i, conv in enumerate(convs):
                convs[i] = nn.Sequential(
                    conv,
                    nn.Dropout3d()
                )

        # Create double convolution block.
        self.double_conv = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: bool = False):
        """
        args:
            in_channels: the number of channels into the module.
            out_channels: the number of channels out of the module.
        kwargs:
            dropout: include dropout layer.
        """
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: bool = False):
        """
        args:
            in_channels: the number of channels into the module.
            out_channels: the number of channels out of the module.
        kwargs:
            dropout: include dropout layer.
        """
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_res: torch.Tensor) -> torch.Tensor:
        """
        returns: tensor result of forward pass through module.
        args:
            x: the input tensor.
            x_res: the residual input tensor
        """
        # Get upsampled features.
        x = self.upsample(x)

        # Concatenate upsampled and residual features.
        if x_res.shape != x.shape:
            raise ValueError(f"Residual shape '{x_res.shape}', doesn't match shape of upsampled features '{x.shape}'. Input volume dimensions must be divisible by 16.")
        x = torch.cat((x, x_res), dim=1)

        # Perform double convolution.
        x = self.double_conv(x)

        return x

class OutConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int):
        """
        args:
            in_channels: the number of channels into the module.
            out_channels: the number of channels out of the module.
        """
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SingleChannelUNet(nn.Module):
    def __init__(
        self,
        dropout: bool = False):
        """
        args:
            dropout: include network dropout.
        """
        super().__init__()
        self.first = DoubleConv(1, 64, dropout=dropout)
        self.down1 = Down(64, 128, dropout=dropout)
        self.down2 = Down(128, 256, dropout=dropout)
        self.down3 = Down(256, 512, dropout=dropout)
        self.down4 = Down(512, 1024, dropout=dropout)
        self.up1 = Up(1024, 512, dropout=dropout)
        self.up2 = Up(512, 256, dropout=dropout)
        self.up3 = Up(256, 128, dropout=dropout)
        self.up4 = Up(128, 64, dropout=dropout)
        self.out = OutConv(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        return x

    def num_params(self) -> int:
        """
        returns: the number of network parameters, i.e. weights and biases.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
