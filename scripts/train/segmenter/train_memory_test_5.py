import numpy as np
import os
from tqdm import tqdm

SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

# Run experiments.
name = 'memory-test-mod-2'
n_voxelses = np.arange(0, 10e7, 0.1e7, dtype=int)
ckpt_library = 'baseline'
ckpt_mode = ''
halve_channels = False
double_groups = False
n_split_channels = 2
n_ckpts = 20
# record_times = [30] * 10 + [45] * (len(n_voxelses) - 10)
record_time = 15

for n_voxels in n_voxelses:
    command = f"""
    python {SCRIPT_DIR}/train/segmenter/train_memory_test.py \
        --name={name} \
        --n_voxels={n_voxels} \
        --ckpt_library={ckpt_library} \
        --ckpt_mode={ckpt_mode} \
        --double_groups={double_groups} \
        --halve_channels={halve_channels} \
        --n_ckpts={n_ckpts} \
        --n_split_channels={n_split_channels} \
        --record_time={record_time}
    """
    print(command)
    os.system(command)
