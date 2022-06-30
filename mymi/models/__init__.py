import os
import torch
from typing import List, Tuple

from mymi import config
from mymi.reporting.models import load_model_manifest

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

def replace_checkpoint_alias(
    name: str,
    run: str,
    ckpt: str,
    use_model_manifest: bool = False) -> Tuple[str, str, str]:
    if ckpt == 'BEST': 
        if use_model_manifest:
            man_df = load_model_manifest()
            ckpts = man_df[(man_df.name == name) & (man_df.run == run) & (man_df.checkpoint != 'last')].sort_values('checkpoint')
            assert len(ckpts) >= 1
            ckpt = ckpts.iloc[-1].checkpoint
        else:
            ckptspath = os.path.join(config.directories.models, name, run)
            ckpts = list(sorted([c.replace('.ckpt', '') for c in os.listdir(ckptspath)]))
            ckpt = ckpts[-1]
    else:
        raise ValueError(f"Checkpoint alias '{ckpt}' not recognised.")
    return (name, run, ckpt)
