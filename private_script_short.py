from functools import reduce
import json
import numpy as np
import torch
from torch import nn

image_shape = (377, 370, 251)
n_channels = 64
input_shape = (1, n_channels, *image_shape)
n_voxels = reduce(np.multiply, input_shape)
int_max = 2 ** 31 - 1
print(n_voxels <= int_max)
device = torch.device('cuda:0')
input = torch.rand(input_shape)

input = input.half().to(device)
print('input GB: ', input.numel() * input.element_size() / 1e9)
layer = nn.Conv3d(in_channels=n_channels, out_channels=32, groups=4, kernel_size=3, stride=1, padding=1).half().to(device)
output = layer(input)
print('max alloc. (GB): ', torch.cuda.max_memory_allocated() / 1e9)
print('max res. (GB): ', torch.cuda.max_memory_reserved() / 1e9)

