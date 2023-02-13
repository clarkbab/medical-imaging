import numpy as np
from typing import List, Tuple

LEVELS = [
    (0, 5),
    (6, 12),
    (13, 19),
    (20, 26),
    (27, 33),
    (34, 40),
    (41, 47),
    (48, 54),
    (55, 63)
]

def get_checkpoint_sizes(
    n_layers: int,
    n_ckpts: int) -> List[int]:
    # Get number of layers per checkpoint.
    ckpt_sizes = []
    while n_layers > 0:
        ckpt_size = float(n_layers) / n_ckpts  
        if ckpt_size < 2:
            raise ValueError(f"'n_ckpts' must be half 'n_layers' or less. Got '{n_ckpts}' and '{n_layers}'.")
        if ckpt_size == int(ckpt_size):
            ckpt_sizes += [int(ckpt_size)] * n_ckpts
            n_layers = 0
        else:
            ckpt_size = int(np.floor(ckpt_size))
            ckpt_sizes.append(ckpt_size)
            n_layers -= ckpt_size
            n_ckpts -= 1

    # Sort so largest checkpoints are in the centre - this is because we
    # want more fine-grained checkpoints for the upper levels (maybe?). 
    n_min = ckpt_sizes.count(np.min(ckpt_sizes))
    n_roll = int(np.floor(n_min / 2))
    ckpt_sizes = list(np.roll(ckpt_sizes, -n_roll))

    return ckpt_sizes

def get_checkpoints(
    n_layers: int,
    n_ckpts: int) -> List[Tuple[int, int]]:
    ckpt_sizes = get_checkpoint_sizes(n_layers, n_ckpts)

    # Create checkpoints.
    ckpts = []
    layer = 0
    for ckpt_size in ckpt_sizes:
        ckpt = (layer, layer + ckpt_size - 1)
        ckpts.append(ckpt)
        layer += ckpt_size
        
    return ckpts

# Assumes that splits don't occur across a level.
def get_level_checkpoint_sizes(n_ckpts: int):
    pass