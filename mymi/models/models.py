import json
import os
import re
import torch
from typing import *

from mymi import config
from mymi import logging
from mymi.reporting.models import load_model_manifest
from mymi.typing import *
from mymi.utils import *

LAST_CKPT_REGEX = r'^last(-v([0-9]+))?$'

def load_model(
    module: torch.nn.Module,
    project: str,
    model: ModelName,
    ckpt: ModelCheckpoint,
    device: Optional[torch.device] = None,
    state: Literal['train', 'eval'] = 'eval',
    **kwargs) -> Union[torch.nn.Module, torch.device, Dict[str, float]]:
    # Create model.
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    loaded_model = module(**kwargs)

    # Load model state.
    ckpt = replace_ckpt_alias(project, model, ckpt)
    filepath = os.path.join(config.directories.models, project, model, f'{ckpt}.ckpt')
    ckpt_data = torch.load(filepath, map_location=device)
    logging.info(f"Loading model {project}/{model} ({module}) from epoch={ckpt_data['epoch']}.")
    loaded_model.load_state_dict(ckpt_data['model'])
    loaded_model.to(device)
    loaded_model.train() if state == 'train' else loaded_model.eval()

    # Return other info for rest of code to use.
    del ckpt_data['model']

    return loaded_model, device, ckpt_data

def replace_ckpt_alias(
    project: str,
    model: str,
    ckpt: str,
    ckpt_version: Optional[int] = None,
    use_manifest: bool = False) -> str:
    ckpt = ckpt.lower()
    if ckpt.endswith('.ckpt'):
        ckpt = ckpt.replace('.ckpt', '')

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
            ckpts_path = os.path.join(config.directories.models, project, model)
            if not os.path.exists(ckpts_path):
                raise ValueError(f"No model {project}/{model} exists.")
            ckpts = [c for c in os.listdir(ckpts_path) if '.ckpt' in c and c != 'last.ckpt']
            ckpts = list(sorted([c.replace('.ckpt', '') for c in ckpts]))
            ckpt = ckpts[-1]

    elif ckpt == 'last':
        # Get specific version of 'last' checkpoint.
        if ckpt_version is not None:
            if ckpt_version > 0:
                ckpt = f'last-v{ckpt_version}'
                filepath = os.path.join(config.directories.models, *model[:2], f'{ckpt}.ckpt')
                if not os.path.exists(filepath):
                    raise ValueError(f"No '{ckpt}' checkpoint exists for model '{model[0]}', run '{model[1]}'. Filepath: {filepath}.")

        # Get latest 'last' checkpoint.
        else:
            filepath = os.path.join(config.directories.models, project, model)
            if not os.path.exists(filepath):
                raise ValueError(f"No model {project}/{model} exists.")
            ckpts = os.listdir(filepath)
            ckpts = [c.replace('.ckpt', '') for c in ckpts]
            last_ckpts = list(sorted([c for c in ckpts if re.match(LAST_CKPT_REGEX, c) is not None]))
            if len(last_ckpts) == 0:
                raise ValueError(f"No 'last' checkpoint exists for model {project}/{model}.")

            # Find the latest 'last' checkpoint.
            ckpt = last_ckpts[-1]

    return ckpt
