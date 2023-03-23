import numpy as np
import os
from tqdm import tqdm

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

# Run experiments.
name = 'memory-test'
n_voxelses = np.arange(0, 1e7, 0.1e7, dtype=int)
ckpt_library = 'baseline'
ckpt_mode = ''
halve_channels = False

for n_voxels in n_voxelses:
    command = f"""
    python {SCRIPT_DIR}/train/segmenter/train_memory_test.py \
        --name={name} \
        --n_voxels={n_voxels} \
        --ckpt_library={ckpt_library} \
        --ckpt_mode={ckpt_mode} \
        --halve_channels={halve_channels}
    """
    print(command)
    os.system(command)
