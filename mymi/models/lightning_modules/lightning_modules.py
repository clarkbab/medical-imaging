import os
import re
from typing import *

from mymi import config
from mymi.reporting.models import load_model_manifest
from mymi.typing import *

LAST_CKPT_REGEX = r'^last(-v([0-9]+))?$'

def replace_ckpt_alias(
    model: ModelName,
    ckpt_version: Optional[int] = None,
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
            filepath = os.path.join(config.directories.models, *model[:2])
            if not os.path.exists(filepath):
                raise ValueError(f"No run '{model[1]}' exists for model '{model[0]}'. Filepath: {filepath}.")
            ckpts = os.listdir(filepath)
            ckpts = [c.replace('.ckpt', '') for c in ckpts]
            last_ckpts = list(sorted([c for c in ckpts if re.match(LAST_CKPT_REGEX, c) is not None]))
            if len(last_ckpts) == 0:
                raise ValueError(f"No 'last' checkpoint exists for model '{model[0]}', run '{model[1]}'. Filepath: {filepath}.")

            # Find the latest 'last' checkpoint.
            ckpt = last_ckpts[-1]

    return (*model[:2], ckpt)
