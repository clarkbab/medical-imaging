import os
from threading import Thread
from time import sleep
import torch
from torch import nn

from mymi.reporting.gpu_usage import record_gpu_usage

n_channels = 64
name = f'testing-{n_channels}'
input_shape = (1, n_channels, 370, 370, 250)
device = torch.device('cuda:0')
input = torch.rand(input_shape)

# Kick off GPU memory recording.
thread = Thread(target=record_gpu_usage, args=(name, 10, 1e-3))
thread.start()
# sleep(3)

input = input.half().to(device)
input_GB = input.numel() * input.element_size() / 1e9
print('input GB: ', input_GB)
input.requires_grad = False
layer = nn.Conv3d(in_channels=n_channels, out_channels=32, kernel_size=3, stride=1, padding=1).half().to(device)
for i, param in enumerate(layer.parameters()):
    param_GB = param.numel() * param.element_size() / 1e9
    print(f'param_{i} GB: {param_GB}')
    param.requires_grad = False
output = layer(input)

torch.cuda.empty_cache()
