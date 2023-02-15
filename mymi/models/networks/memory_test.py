import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.checkpoint import checkpoint
from fairscale.nn.checkpoint import checkpoint_wrapper
from typing import Dict, List, Literal

from mymi.checkpointing import get_checkpoints, get_level_checkpoints

PRINT_ENABLED = True

class MemoryTestSubmodule(nn.Module):
    def __init__(
        self,
        name: str,
        layers: List[nn.Module],
        residual_outputs: List[int] = [],
        residual_output_layers: Dict[int, nn.Module] = {},
        residual_inputs: List[int] = [],
        residual_input_layers: Dict[int, nn.Module] = {},
        print_enabled: bool = PRINT_ENABLED) -> None:
        super().__init__()
        self.__layers = nn.ParameterList(layers)
        self.__name = name
        self.__print_enabled = print_enabled
        self.__residual_outputs = residual_outputs
        self.__residual_inputs = residual_inputs
        self.__residual_output_layer_keys = residual_output_layers.keys()
        self.__residual_output_layer_values = nn.ParameterList(residual_output_layers.items())
        self.__residual_input_layer_keys = residual_input_layers.keys()
        self.__residual_input_layer_values = nn.ParameterList(residual_input_layers.items())

    def __pad_upsampled(self, x, shape):
        if x.shape != shape: 
            n_axes = len(x.shape)
            padding = np.zeros((n_axes, 2), dtype='uint8')
            for axis in range(n_axes):
                diff = shape[axis] - x.shape[axis]
                if diff > 0:
                    padding[axis] = np.floor([diff / 2, (diff + 1) / 2])
            padding = tuple(np.flip(padding, axis=0).flatten())
            x = pad(x, padding)
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
                x = torch.cat((x, x_res_i), dim=1)

            # Execute layer.
            x = layer(x)

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

class LayerWrapper(nn.Module):
    def __init__(
        self,
        layer: nn.Module,
        name: str,
        print_enabled: bool = PRINT_ENABLED) -> None:
        super().__init__()
        self.__print_enabled = print_enabled
        self.__layer = layer
        self.__name = name

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
    
class MemoryTest(nn.Module):
    def __init__(
        self,
        n_output_channels: int,
        ckpt_library: Literal['baseline', 'ckpt-pytorch', 'ckpt-fairscale', 'ckpt-fairscale-offload'] = 'baseline',
        ckpt_mode: Literal['', '-level'] = '',
        n_ckpts: int = 1,
        n_gpus: int = 0) -> None:
        super().__init__()
        self.__n_gpus = n_gpus

        # Set mode flags. 
        if ckpt_library == 'baseline':
            self.__use_pytorch_ckpt = False
            self.__use_fairscale_ckpt = False
            self.__use_fairscale_cpu_offload = False
        if ckpt_library == 'ckpt-pytorch':
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
        self.__layers = nn.ParameterList()
        residuals = [
            [5, 35],
            [12, 42],
            [19, 49],
            [26, 56]
        ] 
        residual_halves = [
            # [5, 35]
        ]
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
        in_channels = 1
        out_channels = n_features
        self.__layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
        self.__layers.append(nn.InstanceNorm3d(out_channels))
        self.__layers.append(nn.ReLU())
        self.__layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
        self.__layers.append(nn.InstanceNorm3d(out_channels))
        self.__layers.append(nn.ReLU())

        # Add downsampling levels. 
        self.__n_down_levels = 4
        for i in range(self.__n_down_levels):
            in_channels = 2 ** i * n_features
            out_channels = 2 ** (i + 1) * n_features
            self.__layers.append(nn.MaxPool3d(kernel_size=2))
            self.__layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
            self.__layers.append(nn.InstanceNorm3d(out_channels))
            self.__layers.append(nn.ReLU())
            self.__layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_0))
            self.__layers.append(nn.InstanceNorm3d(out_channels))
            self.__layers.append(nn.ReLU())

        # Add upsampling levels.
        self.__n_up_levels = 4
        for i in range(self.__n_up_levels):
            in_channels = 2 ** (self.__n_up_levels - i) * n_features
            out_channels = 2 ** (self.__n_up_levels - i - 1) * n_features
            self.__layers.append(nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2).to(self.__device_1))
            self.__layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_2))
            self.__layers.append(nn.InstanceNorm3d(out_channels))
            self.__layers.append(nn.ReLU())
            self.__layers.append(nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(self.__device_3))
            self.__layers.append(nn.InstanceNorm3d(out_channels))
            self.__layers.append(nn.ReLU())

        # Add final layers.
        in_channels = n_features
        out_channels = n_output_channels
        self.__layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self.__layers.append(nn.Softmax(dim=1))

        # Wrap each layer in a print layer to test.
        self.__layers = nn.ParameterList([LayerWrapper(l, str(i)) for i, l in enumerate(self.__layers)])

        # Get checkpoint locations.
        n_layers = len(self.__layers)
        assert n_layers == 64
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
                if res_output >= start_layer and res_output <= end_layer:
                    layer = nn.Conv3d(in_channels=0, out_channels=0, kernel_size=3, stride=1, padding=1).to('cuda:0')
                    residual_output_layers[res_output - start_layer] = layer
                if res_input >= start_layer and res_input <= end_layer:
                    layer = nn.Conv3d(in_channels=0, out_channels=0, kernel_size=3, stride=1, padding=1).to('cuda:0')
                    residual_input_layers[res_input - start_layer] = layer

            layers = self.__layers[start_layer:end_layer + 1]
            module = MemoryTestSubmodule(str(i), layers, residual_outputs=residual_outputs, residual_output_layers=residual_output_layers, residual_inputs=residual_inputs, residual_input_layers=residual_input_layers)
            if self.__use_fairscale_ckpt:
                module = checkpoint_wrapper(module, offload_to_cpu=self.__use_fairscale_cpu_offload)
            self.__submodules.append(module)

    @property
    def layers(self) -> List[nn.Module]:
        return self.__layers

    @property
    def submodules(self) -> List[nn.Module]:
        return self.__submodules

    def forward(self, x):
        x_res = []
        for module in self.__submodules:
            if self.__use_pytorch_ckpt:
                dummy_arg = torch.Tensor()
                dummy_arg.requires_grad = True
                x, *x_res = checkpoint(module, x, *x_res, dummy_arg=dummy_arg, use_reentrant=False)
            else:
                x, *x_res = module(x, *x_res)
        assert len(x_res) == 0
        return x