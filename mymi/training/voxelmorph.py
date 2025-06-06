import os
import shutil
import subprocess
import sys
from typing import *

VXM_PATH = os.path.join(os.environ['CODE'], 'voxelmorph')
os.environ['VXM_BACKEND'] = 'pytorch'
sys.path.append(VXM_PATH)

from mymi import config
from mymi.datasets import TrainingDataset
from mymi import logging
from mymi.typing import *

def train_voxelmorph(
    dataset: str,
    model: str,
    pad_shape: Optional[Size3D] = None) -> None:
    set = TrainingDataset(dataset)
    index_path = os.path.join(set.path, 'vxm-index.txt')
    model_path = os.path.join(config.directories.models, 'voxelmorph', f'{dataset}-{model}')
    if os.path.exists(model_path):
        shutil.rmtree(model_path)    # Clear out old runs.

    command = [
        'python', f'{VXM_PATH}/scripts/torch/train.py',
        '--gpu', '0',
        '--img-list', index_path,
        '--int-steps', '0',   # DVF values exploded during training.
        '--model-dir', model_path,
    ]
    if pad_shape is not None:
        command += ['--pad-shape'] + [str(p) for p in pad_shape]
    logging.info(command)
    subprocess.run(command)
