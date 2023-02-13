import os
from tqdm import tqdm

MODES = [
    # 'baseline',
    # 'ckpt-pytorch',
    'ckpt-fairscale',
    'ckpt-fairscale-offload'
]
MONITOR_TIMES = {
    'baseline': 30,
    'ckpt-pytorch': 30,
    'ckpt-fairscale': 30,
    'ckpt-fairscale-offload': 60
}
N_CKPTSES = list(range(1, 33))
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

# Run experiments.
for mode in tqdm(MODES):
    # Run with evenly spaced checkpoints.
    for n_ckpts in tqdm(N_CKPTSES, leave=False):
        name = f'memory-test-small-{mode}-n_ckpts-{n_ckpts}'
        command = f"""
python {SCRIPT_DIR}/train/segmenter/train_memory_test.py \
    --mode={mode} \
    --name={name} \
    --monitor_time={MONITOR_TIMES[mode]} \
    --n_ckpts={n_ckpts}
"""
        print(command)
        os.system(command)
