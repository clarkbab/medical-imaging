import os
from tqdm import tqdm

CKPT_LIBRARIES = [
    'baseline',
    'ckpt-pytorch',
    'ckpt-fairscale',
    'ckpt-fairscale-offload'
]
INPUT_MODES = [
    "-large",
    # "",
    # "-small",
    # "-xsmall"
]
CKPT_MODES = [
    # "",
    "-level"
]
MIN_CKPTS = {
    "": 1,
    "-level": 1
}
MAX_CKPTS = {
    "": 32,
    "-level": 3
}
MONITOR_TIMES = {
    '-large': {
        'baseline': 30,
        'ckpt-pytorch': 35,
        'ckpt-fairscale': 35,
        'ckpt-fairscale-offload': 90
    },
    '': {
        'baseline': 30,
        'ckpt-pytorch': 40,
        'ckpt-fairscale': 40,
        'ckpt-fairscale-offload': 70
    },
    '-small': {
        'baseline': 20,
        'ckpt-pytorch': 25,
        'ckpt-fairscale': 25,
        'ckpt-fairscale-offload': 70
    },
    '-xsmall': {
        'baseline': 15,
        'ckpt-pytorch': 15,
        'ckpt-fairscale': 15,
        'ckpt-fairscale-offload': 35
    }
}
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

# Run experiments.
for ckpt_library in tqdm(CKPT_LIBRARIES):
    for ckpt_mode in tqdm(CKPT_MODES, leave=False):
        n_ckptses = list(range(MIN_CKPTS[ckpt_mode], MAX_CKPTS[ckpt_mode] + 1))
        for input_mode in tqdm(INPUT_MODES, leave=False):
            # Run with evenly spaced checkpoints.
            for n_ckpts in tqdm(n_ckptses, leave=False):
                command = f"""
python {SCRIPT_DIR}/train/segmenter/train_memory_test.py \
    --ckpt_library={ckpt_library} \
    --ckpt_mode={ckpt_mode} \
    --input_mode={input_mode} \
    --n_ckpts={n_ckpts} \
    --monitor_time={MONITOR_TIMES[input_mode][ckpt_library]}
"""
                print(command)
                os.system(command)
