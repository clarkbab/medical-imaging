from fire import Fire
import os
import sys
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(root_dir)

from mymi import config

def print_epoch(
    model: str,
    run: str,
    checkpoint: str = 'last') -> None:
    # Load checkpoint state.
    model = (model, run, f'{checkpoint}.ckpt')
    path = os.path.join(config.directories.models, *model)
    state = torch.load(path, map_location=torch.device('cpu'))

    print(f"""
Model: {model}
Epoch: {state['epoch']}
""")

Fire(print_epoch)
