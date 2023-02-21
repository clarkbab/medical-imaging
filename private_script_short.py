import json
import os
import torch
from torch import nn

# print('PYTORCH_CUDA_ALLOC_CONF', os.environ['PYTORCH_CUDA_ALLOC_CONF'])

# x_dim = 361   # works - max memory ~12.9GB.
x_dim = 362     # breaks - tries to allocate ~110GiB.
input_shape = (1, 64, x_dim, 370, 251)
device = torch.device('cuda:0')
input = torch.rand(input_shape)

input = input.half().to(device)
print('input GB: ', input.numel() * input.element_size() / 1e9)
layer = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1).half().to(device)
try:
    output = layer(input)
except torch.cuda.OutOfMemoryError:
    print('OOMed!!!')

print('max alloc. (GB): ', torch.cuda.max_memory_allocated() / 1e9)
print('max res. (GB): ', torch.cuda.max_memory_reserved() / 1e9)
filepath = 'memory-stats.json'
with open(filepath, 'w') as f:
    f.write(json.dumps(torch.cuda.memory_stats()))

