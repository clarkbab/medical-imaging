from functools import partial, reduce
import fairscale
import numpy as np
import torch
from typing import *

from mymi.misc.checkpointing import get_checkpoints, get_level_checkpoints
from mymi import logging
from mymi.utils import *

CUDA_INT_MAX = 2 ** 31 - 1
PRINT_ENABLED = False

class Submodule(torch.nn.Module):
    def __init__(
        self,
        name: str,
        layers: List[torch.nn.Module],
        residual_outputs: List[int] = [],
        residual_output_layers: Dict[int, torch.nn.Module] = {},
        residual_inputs: List[int] = [],
        residual_input_layers: Dict[int, torch.nn.Module] = {},
        print_enabled: bool = PRINT_ENABLED) -> None:
        super().__init__()
        self.__layers = torch.nn.ParameterList(layers)
        self.__name = name
        self.__print_enabled = print_enabled
        self.__residual_outputs = residual_outputs
        self.__residual_inputs = residual_inputs
        self.__residual_output_layer_keys = list(residual_output_layers.keys())
        self.__residual_output_layer_values = torch.nn.ParameterList(residual_output_layers.values())
        self.__residual_input_layer_keys = list(residual_input_layers.keys())
        self.__residual_input_layer_values = torch.nn.ParameterList(residual_input_layers.values())

    @property
    def name(self) -> str:
        return self.__name

    def __pad_upsampled(self, x, shape):
        if x.shape != shape: 
            n_axes = len(x.shape)
            padding = np.zeros((n_axes, 2), dtype='uint8')
            for axis in range(n_axes):
                diff = shape[axis] - x.shape[axis]
                if diff > 0:
                    padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
            padding = tuple(np.flip(padding, axis=0).flatten())
            x = torch.nn.functional.pad(x, padding)
        return x

    def forward(self, x, *x_res, dummy_arg=None):
        if self.__print_enabled:
            print(f'=== submodule ({self.__name}) ===')
            print(f'res_outputs: {self.__residual_outputs}')
            print(f'res_inputs: {self.__residual_inputs}')
            print(f'id(x_res): {id(x_res)}')
            print(f'len(x_res): {len(x_res)}')
            print(f'shape(x_res): {[xr.shape for xr in x_res]}')
            print(f'x.requires_grad: ', x.requires_grad)
            print(f'x_res.requires_grad: ', [xr.requires_grad for xr in x_res])

        for i, layer in enumerate(self.__layers):
            # Upsample and concat input with residual.
            if i in self.__residual_inputs:
                # Remove element from 'x_res' but not in-place as this
                # will break 'x_res' on second forward pass with checkpointing.
                x_res_i = x_res[-1]
                x_res = [xr for j, xr in enumerate(x_res) if j != len(x_res) - 1]
                x = self.__pad_upsampled(x, x_res_i.shape) 

                # Apply pre-input layer.
                if i in self.__residual_input_layer_keys:
                    x = self.__residual_input_layer_values[self.__residual_input_layer_keys.index(i)](x)

                # Concatenate upsampled and residual features.
                if self.__print_enabled:
                    print(f"(layer {layer.name}) pre-concat shape: {x.shape}")
                    print(f"(layer {layer.name}) residual shape: {x_res_i.shape}")
                x = torch.cat((x, x_res_i), dim=1)
                if self.__print_enabled:
                    print(f"(layer {layer.name}) concat shape: {x.shape}")
                    print(f"(layer {layer.name}) concat size (GB): {x.numel() * x.element_size() / 1e9}")

            # Execute layer.
            n_voxels = reduce(np.multiply, x.shape)
            if n_voxels > CUDA_INT_MAX:
                logging.error(f"Feature map of size '{x.shape}' (n_voxels={n_voxels}) has more voxels than 'CUDA_INT_MAX' ({CUDA_INT_MAX}) voxels.")
            # x_in = x.detach().cpu().numpy()
            x = layer(x)
            # if torch.any(torch.isnan(x)).item():
            #     print(f"Layer {layer.name} has nan output.")
            #     import os
            #     from mymi import config
            #     filepath = os.path.join(config.directories.temp, f'{i}-input.npy')
            #     np.save(filepath, x_in)
            #     filepath = os.path.join(config.directories.temp, f'{i}-output.npy')
            #     np.save(filepath, x.detach().cpu().numpy())
            #     filepath = os.path.join(config.directories.temp, f'{i}-layer.ckpt')
            #     torch.save(layer.state_dict(), filepath)

            # Add 'x' to 'x_res'.
            if i in self.__residual_outputs:
                x_res_new = x
                # Apply pre-output layer.
                if i in self.__residual_output_layer_keys:
                    x_res_new = self.__residual_output_layer_values[self.__residual_output_layer_keys.index(i)](x_res_new)
                x_res = tuple(list(x_res) + [x_res_new])     # Note that 'x_res += [x]' is still an in-place operation!

        if self.__print_enabled:
            print(f'id(x_res): {id(x_res)}')
            print(f'len(x_res): {len(x_res)}')
            print(f'shape(x_res): {[xr.shape for xr in x_res]}')
            print(f'x.requires_grad: ', x.requires_grad)
            print(f'x_res.requires_grad: ', [xr.requires_grad for xr in x_res])
            print('=================')

        return x, *x_res

class HybridOutput(torch.nn.Module):
    def __init__(
        self,
        sigmoid_channels: List[int] = [],
        softmax_channels: List[int] = []) -> None:
        if len(softmax_channels) == 1:
            raise ValueError(f"'softmax_channels' must be 0 or greater than 1, otherwise channel output will always be 1.")
        self.__sigmoid_channels = sigmoid_channels
        self.__softmax_channels = softmax_channels

    def forward(self, x):
        n_channels = len(self.__sigmoid_channels) + len(self.__softmax_channels)
        if x.shape[1] != n_channels:
            raise ValueError(f"'x' channel dimension ({x.shape[1]}) should equal number of requested channels ({n_channels}).")

        # Apply sigmoid.
        x_sig = x[:, self.__sigmoid_channels]
        x_sig = torch.nn.functional.sigmoid(x_sig)

        # Apply softmax.
        x_soft = x[:, self.__softmax_channels]
        x_soft = torch.nn.functional.softmax(x_soft, dim=1)

        # Create output.
        x = torch.zeros_like(x)
        x[:, self.__sigmoid_channels] = x_sig
        x[:, self.__softmax_channels] = x_soft

        return x

class SplitConv3D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        n_split_channels: int = 2,
        groups: int = 1,
        padding: int = 1,
        stride: int = 1) -> None:
        super().__init__()
        assert in_channels % n_split_channels == 0
        assert out_channels % n_split_channels == 0

        self.__in_channels = in_channels
        self.__split_size = self.__in_channels // n_split_channels

        # Create split layers.
        in_channels = self.__in_channels // n_split_channels
        out_channels = out_channels // n_split_channels
        layers = []
        for _ in range(n_split_channels):
            layer = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            layers.append(layer)
        self.__layers = torch.nn.ParameterList(layers)

    def forward(self, x) -> torch.Tensor:
        # Split tensor along channel dimension.
        assert x.shape[1] == self.__in_channels
        xs = list(torch.split(x, self.__split_size, dim=1))

        # Process each split.
        for i, x in enumerate(xs):
            xs[i] = self.__layers[i](x)

        # Concatenate results.
        x = torch.cat(xs, dim=1)

        return x

class LayerWrapper(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        name: str,
        print_enabled: bool = PRINT_ENABLED) -> None:
        super().__init__()
        self.__print_enabled = print_enabled
        self.__layer = layer
        self.__name = name

    @property
    def layer(self) -> torch.nn.Module:
        return self.__layer

    @property
    def name(self) -> str:
        return f"{self.__name} ({self.__layer})"

    @property
    def out_channels(self) -> Optional[int]:
        if hasattr(self.__layer, 'out_channels'):
            return self.__layer.out_channels
        else:
            return None

    def forward(self, x) -> torch.Tensor:
        if self.__print_enabled:
            print(f'>>> layer: {self.__name} ({type(self.__layer)}) >>>')
            print(f'input shape: {x.shape}')
            print(f'input type: {x.dtype}')

        x = self.__layer(x)

        if self.__print_enabled:
            print(f'output shape: {x.shape}')
            print(f'output type: {x.dtype}')
            print('>>>>>>>>>>>>>>>>>')

        return x
    
class UNet3D(torch.nn.Module):
    def __init__(
        self,
        n_output_channels: int,
        ckpt_library: Literal['baseline', 'ckpt-pytorch', 'ckpt-fairscale', 'ckpt-fairscale-offload'] = 'baseline',
        ckpt_mode: Literal['', '-level'] = '',
        devices: Union[torch.device, List[torch.device]] = torch.device('cpu'),
        double_groups: bool = False,
        halve_channels: bool = False,
        n_ckpts: int = 22,
        n_features: int = 32,   # S=16, M=32, L=64, XL=128
        n_input_channels: int = 1,
        n_split_channels: int = 2,
        use_affine_norm: bool = True,
        use_init: bool = False,
        use_single_downsample: bool = False,
        use_softmax: bool = True,
        use_small_output_params: bool = False,
        **kwargs) -> None:
        super().__init__()
        devices = arg_to_list(devices, torch.device)
        assert not ((double_groups and halve_channels) or (double_groups and n_split_channels > 1))

        # Set mode flags. 
        if ckpt_library == 'baseline':
            self.__use_pytorch_ckpt = False
            self.__use_fairscale_ckpt = False
            self.__use_fairscale_cpu_offload = False
        elif ckpt_library == 'ckpt-pytorch':
            self.__use_pytorch_ckpt = True
            self.__use_fairscale_ckpt = False
            self.__use_fairscale_cpu_offload = False
        elif ckpt_library == 'ckpt-fairscale':
            self.__use_pytorch_ckpt = False
            self.__use_fairscale_ckpt = True
            self.__use_fairscale_cpu_offload = False
        elif ckpt_library == 'ckpt-fairscale-offload':
            self.__use_pytorch_ckpt = False
            self.__use_fairscale_ckpt = True
            self.__use_fairscale_cpu_offload = True
        else:
            raise ValueError(f"'ckpt_library={ckpt_library}' not recognised.")

        # Assign devices based on number of GPUs.
        assert len(devices) in (1, 2, 4)
        if len(devices) == 1:
            self.__device_0, self.__device_1, self.__device_2, self.__device_3 = 4 * devices
        elif len(devices) == 2:
            self.__device_0, self.__device_2, self.__device_1, self.__device_3 = 2 * devices
        else:
            self.__device_0, self.__device_1, self.__device_2, self.__device_3 = devices

        # Define layers.
        self.__layers = torch.nn.ParameterList()
        residuals = [
            [5, 56],
            [12, 49],
            [19, 42],
            [26, 35]
        ] 
        # residuals = [
        #     [3, 40],
        #     [8, 35],
        #     [13, 30],
        #     [18, 25]
        # ]
        if halve_channels:
            residual_halves = [     # Halve the number of channels for residual output and upsampled input.
                [5, 56]
            ]
            # residual_halves = [
            #     [3, 40]
            # ]
        else:
            residual_halves = []
        levels = [
            (0, 5),
            (6, 12),
            (13, 19),
            (20, 26),
            (27, 33),
            (34, 40),
            (41, 47),
            (48, 54),
            (55, 63)
        ]

        # Add first level.
        in_channels = n_input_channels
        out_channels = n_features
        self.__layers.append(torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
        self.__layers.append(torch.nn.InstanceNorm3d(out_channels, affine=use_affine_norm))
        self.__layers.append(torch.nn.ReLU())
        self.__layers.append(torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
        self.__layers.append(torch.nn.InstanceNorm3d(out_channels, affine=use_affine_norm))
        self.__layers.append(torch.nn.ReLU())

        # Add downsampling levels. 
        self.__n_down_levels = 4
        for i in range(self.__n_down_levels):
            in_channels = 2 ** i * n_features
            out_channels = 2 ** (i + 1) * n_features
            if not use_single_downsample or i == 0:
                self.__layers.append(torch.nn.MaxPool3d(kernel_size=2))
            self.__layers.append(torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
            self.__layers.append(torch.nn.InstanceNorm3d(out_channels, affine=use_affine_norm))
            self.__layers.append(torch.nn.ReLU())
            self.__layers.append(torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
            self.__layers.append(torch.nn.InstanceNorm3d(out_channels, affine=use_affine_norm))
            self.__layers.append(torch.nn.ReLU())

        # Add upsampling levels.
        self.__n_up_levels = 4
        for i in range(self.__n_up_levels):
            in_channels = 2 ** (self.__n_up_levels - i) * n_features
            out_channels = 2 ** (self.__n_up_levels - i - 1) * n_features
            if not use_single_downsample or i == 0:
                self.__layers.append(torch.nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2).to(self.__device_1))
            # Perform some hacks  (largest features maps are too big for cuDNN 32-bit indexing).
            module = torch.nn.Conv3d
            groups = 1
            if i == 3:
                if halve_channels:
                    # Halved the number of residual features maps passed on both residual and upsampling pathways.
                    # See: https://github.com/pytorch/pytorch/issues/95024.
                    in_channels = 32
                if double_groups:
                    # This is equivalent to performing 3D conv operation of half the number of channels in parallel. 
                    # But the implementation doesn't split tensors, so we still have the indexing issue.
                    groups = 2
                if n_split_channels > 1:
                    # Manually split 3D conv operation to reduce tensor size.
                    module = partial(SplitConv3D, n_split_channels=n_split_channels)
            self.__layers.append(module(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=groups).to(self.__device_2))
            self.__layers.append(torch.nn.InstanceNorm3d(out_channels, affine=use_affine_norm))
            self.__layers.append(torch.nn.ReLU())
            self.__layers.append(torch.nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_3))
            self.__layers.append(torch.nn.InstanceNorm3d(out_channels, affine=use_affine_norm))
            self.__layers.append(torch.nn.ReLU())

        # Add final layers.
        in_channels = n_features
        out_channels = n_output_channels
        self.__layers.append(torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        if use_softmax:
            self.__layers.append(torch.nn.Softmax(dim=1))

        # Wrap each layer in a print layer to test.
        self.__layers = torch.nn.ParameterList([LayerWrapper(l, str(i)) for i, l in enumerate(self.__layers)])

        # Get checkpoint locations.
        n_required_layers = 63
        if use_softmax:
            n_required_layers += 1
        n_layers = len(self.__layers)
        assert n_layers == n_required_layers
        if ckpt_mode == '':
            ckpts = get_checkpoints(n_layers, n_ckpts)
        elif ckpt_mode == '-level':
            ckpts = get_level_checkpoints(levels, n_ckpts)

        # Create submodules - can be wrapped with fairscale 'checkpoint_wrapper'.
        self.__submodules = []
        for i, (start_layer, end_layer) in enumerate(ckpts):
            # Add residual inputs/outputs.
            residual_outputs = []
            residual_inputs = []
            for res_output, res_input in residuals:
                if res_output >= start_layer and res_output <= end_layer:
                    residual_outputs.append(res_output - start_layer)
                if res_input >= start_layer and res_input <= end_layer:
                    residual_inputs.append(res_input - start_layer)

            # Add residual input/output additional layers.
            residual_output_layers = {}
            residual_input_layers = {}
            for res_output, res_input in residual_halves:
                if res_output is not None and res_output >= start_layer and res_output <= end_layer:
                    out_layer = res_output
                    in_channels = None
                    while in_channels is None:
                        in_channels = self.__layers[out_layer].out_channels
                        out_layer -= 1
                    out_channels = in_channels // 2
                    res_layer = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0)
                    residual_output_layers[res_output - start_layer] = res_layer
                if res_input is not None and res_input >= start_layer and res_input <= end_layer:
                    out_layer = res_output
                    in_channels = None
                    while in_channels is None:
                        in_channels = self.__layers[out_layer].out_channels
                        out_layer -= 1
                    out_channels = in_channels // 2
                    res_layer = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0)
                    residual_input_layers[res_input - start_layer] = res_layer

            layers = self.__layers[start_layer:end_layer + 1]
            module = Submodule(str(i), layers, residual_outputs=residual_outputs, residual_output_layers=residual_output_layers, residual_inputs=residual_inputs, residual_input_layers=residual_input_layers)
            if self.__use_fairscale_ckpt:
                module = fairscale.nn.checkpoint.checkpoint_wrapper(module, offload_to_cpu=self.__use_fairscale_cpu_offload)
            self.__submodules.append(module)

        # Apply fancy initialisation.
        if use_init:
            logging.info(f"Applying custom parameter initialisation.")
            for m in self.modules():
                if isinstance(m, torch.nn.Conv3d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, torch.nn.InstanceNorm3d):
                    if use_affine_norm:
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.ConvTranspose3d):
                    torch.nn.init.kaiming_normal_(m.weight)

        if use_small_output_params:
            logging.info(f"Setting output layer params to small values.")
            m = self.get_submodule('_UNet3D__layers.62._LayerWrapper__layer')
            torch.nn.init.normal_(m.weight, mean=0, std=1e-5)
            torch.nn.init.constant_(m.bias, 0)

    @property
    def layers(self) -> List[torch.nn.Module]:
        return self.__layers

    @property
    def submodules(self) -> List[torch.nn.Module]:
        return self.__submodules

    def forward(self, x):
        x_res = []
        for i, module in enumerate(self.__submodules):
            if self.__use_pytorch_ckpt:
                dummy_arg = torch.Tensor()
                dummy_arg.requires_grad = True
                x, *x_res = torch.utils.checkpoint(module, x, *x_res, dummy_arg=dummy_arg, use_reentrant=False)
            else:
                x, *x_res = module(x, *x_res)
        assert len(x_res) == 0
        return x
