import os
import shutil
import subprocess

from mymi import config
from mymi.datasets import TrainingDataset

VXM_PATH = "/home/baclark/code/voxelmorph"

def train_voxelmorph(dataset: str) -> None:
    set = TrainingDataset(dataset)
    index_path = os.path.join(set.path, 'vxm-index.txt')
    model_path = os.path.join(config.directories.models, 'voxelmorph', dataset)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)    # Clear out old runs.

    command = [
        'python', f'{VXM_PATH}/scripts/torch/train.py',
        '--gpu', '0',
        '--img-list', index_path,
        '--int-steps', '0',
        '--model-dir', model_path,
    ]
    print(command)
    subprocess.run(command)
