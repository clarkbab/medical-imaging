import os
from tqdm import tqdm

CKPT_LIBRARIES = [
    'baseline',
    'ckpt-pytorch',
    'ckpt-fairscale',
    'ckpt-fairscale-offload'
]
INPUT_MODES = [
    # "",
    # "-small",
    "-xsmall"
]
CKPT_MODES = [
    "",
    # "-level"
]
MONITOR_TIMES = {
    'large': {
        'baseline': 30,
        'ckpt-pytorch': 35,
        'ckpt-fairscale': 35,
        'ckpt-fairscale-offload': 90
    },
    '': {
        'baseline': 25,
        'ckpt-pytorch': 30,
        'ckpt-fairscale': 30,
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
N_CKPTSES = list(range(1, 33))
SCRIPT_DIR="/data/gpfs/projects/punim1413/medical-imaging/scripts"

# Run experiments.
for ckpt_library in tqdm(CKPT_LIBRARIES):
    for ckpt_mode in tqdm(CKPT_MODES, leave=False):
        for input_mode in tqdm(INPUT_MODES, leave=False):
            # Run with evenly spaced checkpoints.
            for n_ckpts in tqdm(N_CKPTSES, leave=False):
                name = f'memory-test{input_mode}-{ckpt_library}-n_ckpts-{n_ckpts}{ckpt_mode}'
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
