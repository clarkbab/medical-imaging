import torch
import torch.nn as nn
from typing import List

def _conv_block(in_channels: int, out_channels: int, use_affine_norm: bool) -> List[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_channels, affine=use_affine_norm),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_channels, affine=use_affine_norm),
        nn.ReLU(inplace=True),
    ]

class BSPUNet2D(nn.Module):
    def __init__(
        self,
        n_features: int = 32,
        n_input_channels: int = 1,
        n_levels: int = 4,
        use_affine_norm: bool = True,
        y_size: int = 768,
    ) -> None:
        super().__init__()

        layers = []

        # First block: no downsampling.
        layers += _conv_block(n_input_channels, n_features, use_affine_norm)

        # Downsampling levels: strided conv halves Y only, channels double.
        for i in range(n_levels):
            in_channels = 2 ** i * n_features
            out_channels = 2 ** (i + 1) * n_features
            # stride=(1, 2): X unchanged, Y halved. padding=(0, 1) preserves exact Y // 2.
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)))
            layers.append(nn.InstanceNorm2d(out_channels, affine=use_affine_norm))
            layers.append(nn.ReLU(inplace=True))
            layers += _conv_block(out_channels, out_channels, use_affine_norm)

        self.__encoder = nn.Sequential(*layers)

        # Fully-connected layer: per-X-position projection from (C_final * Y_reduced) → 2.
        c_final = 2 ** n_levels * n_features
        y_reduced = y_size // (2 ** n_levels)
        self.__fc = nn.Conv1d(c_final * y_reduced, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, X, Y)
        x = self.__encoder(x)
        # x: (B, C_final, X, Y_reduced)

        B, C, X, Y = x.shape
        # Flatten C and Y_reduced, keeping X as the sequence dimension.
        x = x.permute(0, 1, 3, 2).reshape(B, C * Y, X)  # (B, C*Y_reduced, X)

        # Per-position FC: (B, 2, X)
        x = self.__fc(x)

        # Per-channel min-max normalisation to [0, 1].
        x_min = x.amin(dim=2, keepdim=True)
        x_max = x.amax(dim=2, keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)

        return x

