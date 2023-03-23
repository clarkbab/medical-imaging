import numpy as np
import os
import torch

from mymi import config
from mymi.models.networks import MultiUNet3D

def load_tmp():
    names = ['x', 'y', 'y_hat', 'mask', 'weights']
    datas = []
    for name in names:
        filepath = os.path.join(config.directories.temp, f'{name}.npy')
        data = torch.Tensor(np.load(filepath)).cuda()
        datas.append(data)
    filepath = os.path.join(config.directories.temp, 'model.ckpt')
    model = MultiUNet3D(6).cuda()
    model.load_state_dict(torch.load(filepath))
    return [*datas, model]

def load_tmp_layer(i: int, layer: torch.nn.Module):
    filepath = os.path.join(config.directories.temp, f'{i}-input.npy')
    x = torch.Tensor(np.load(filepath)).cuda()
    filepath = os.path.join(config.directories.temp, f'{i}-output.npy')
    y = torch.Tensor(np.load(filepath)).cuda()
    filepath = os.path.join(config.directories.temp, f'{i}-layer.ckpt')
    layer = layer.cuda()
    layer.load_state_dict(torch.load(filepath))
    return x, y, layer

def load_tmp_layers(eyes=list(range(12, 22))):
    datas = []
    for eye in eyes:
        filepath = os.path.join(config.directories.temp, f'{eye}.npy')
        data = np.load(filepath)
        datas.append(data)
    return datas
