import os
from re import match
import torch
from typing import Optional

from mymi import config
from mymi.typing import ModelName

from .architectures import *
from .lightning_modules import *

CKPT_KEYS = [
    'epoch',
    'global_step'
]

def get_localiser(region: str) -> ModelName:
    return (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')

def get_segmenter(
    region: str,
    run: str) -> ModelName:
    return (f'segmenter-{region}', run, 'BEST')

def print_checkpoint(model: ModelName) -> None:
    # Load data.
    checkpoint = f'{model[2]}.ckpt'
    path = os.path.join(config.directories.models, *model[:2], checkpoint)
    data = torch.load(path, map_location=torch.device('cpu'))

    # Print data.
    for k in CKPT_KEYS:
        print(f'{k}: {data[k]}')
