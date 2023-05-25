import os
import torch
from typing import Tuple

from mymi import config
from mymi.reporting.models import load_model_manifest
from mymi.types import ModelName

CHECKPOINT_KEYS = [
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
    for k in CHECKPOINT_KEYS:
        print(f'{k}: {data[k]}')

def replace_checkpoint_alias(
    model: ModelName,
    use_manifest: bool = False) -> ModelName:
    ckpt = model[2].lower()
    if '.ckpt' in ckpt:
        raise ValueError(f"Please do not specify '.ckpt' in model name '{model}'.")

    if ckpt == 'best': 
        if use_manifest:
            # Load model manifest - contains record of all model checkpoint filenames.
            # Remove 'last' and sort to get best checkpoint.
            man_df = load_model_manifest()
            ckpts = man_df[(man_df['model'] == model[0]) & (man_df['run'] == model[1]) & (man_df['ckpt'] != 'last')].sort_values('ckpt', ascending=False)
            if len(ckpts) == 0:
                raise ValueError(f"No record of model '{model}' in model manifest.")
            ckpt = ckpts.iloc[0]['ckpt']
        else:
            # Read model checkpoints from directory.
            # Remove 'last' and sort to get best checkpoint.
            ckpts_path = os.path.join(config.directories.models, *model[:2])
            if not os.path.exists(ckpts_path):
                raise ValueError(f"No run '{model[1]}' found for model '{model[0]}'.")
            ckpts = [c for c in os.listdir(ckpts_path) if '.ckpt' in c and c != 'last.ckpt']
            ckpts = list(sorted([c.replace('.ckpt', '') for c in ckpts]))
            ckpt = ckpts[-1]

    return (*model[:2], ckpt)
