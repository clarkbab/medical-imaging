import os
import torch
from typing import Tuple

from mymi import config

CHECKPOINT_KEYS = [
    'epoch',
    'global_step'
]

def print_checkpoint(model: Tuple[str, str, str]) -> None:
    # Load data.
    checkpoint = f'{model[2]}.ckpt'
    path = os.path.join(config.directories.models, *model[:2], checkpoint)
    data = torch.load(path, map_location=torch.device('cpu'))

    # Print data.
    for k in CHECKPOINT_KEYS:
        print(f'{k}: {data[k]}')
