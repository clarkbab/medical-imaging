import numpy as np
import os
from threading import Thread
from time import sleep
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import SGD
from typing import Literal

# 'libgcc'
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from mymi import config
from mymi import logging
from mymi.losses import DiceWithFocalLoss
from mymi.models.networks import MultiUNet3D
from mymi.reporting.gpu_usage import record_gpu_usage
from mymi.utils import Timer

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_memory_test(
    name: str,
    n_voxels: int,
    ckpt_library: Literal['baseline', 'ckpt-pytorch', 'ckpt-fairscale', 'ckpt-fairscale-offload'] = 'baseline',
    ckpt_mode: Literal['', '-level'] = '',
    double_groups: bool = False,
    halve_channels: bool = False,
    n_channels: int = 5,
    n_ckpts: int = 20,
    n_split_channels: int = 1,
    n_train_steps: int = 11,
    record_interval: float = 1e-3,
    record_time: float = 15) -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logging.arg_log('Running memory test', ('n_voxels', 'ckpt_library', 'ckpt_mode', 'halve_channels', 'n_split_channels'), (n_voxels, ckpt_library, ckpt_mode, halve_channels, n_split_channels))

    # Create name.
    name = f"{name}-input-{n_voxels}"

    # Create model.
    model = MultiUNet3D(n_channels, ckpt_library=ckpt_library, ckpt_mode=ckpt_mode, double_groups=double_groups, halve_channels=halve_channels, n_ckpts=n_ckpts, n_split_channels=n_split_channels).to(device)

    # Create loss function, optimiser and gradient scaler.
    loss_fn = DiceWithFocalLoss()
    optimiser = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = GradScaler()

    # Kick off GPU memory recording.
    thread = Thread(target=record_gpu_usage, args=(name, record_time, record_interval))
    thread.start()
    sleep(2)

    # Create timer.
    timer = Timer(columns={ 'step': int })

    # Run training steps.
    shape = [int(np.cbrt(n_voxels))] * 3
    logging.info(f"Using shape={shape}.")
    for step in range(n_train_steps):
        # Create dummy data.
        input_b = torch.rand(1, 1, *shape).to(device)
        label_b = torch.ones(1, n_channels, *shape, dtype=bool).to(device)
        class_mask_b = torch.ones(1, n_channels, dtype=bool).to(device)
        class_weights_b = torch.ones(1, n_channels).to(device) / n_channels
        
        # Zero all parameter gradients.
        optimiser.zero_grad()

        # Perform training step.
        data = { 'step': step }
        with timer.record(data):
            with autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output_b = model(input_b)
                loss = loss_fn(output_b, label_b, class_mask_b, class_weights_b)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

    # Clear cache - for easy viewing of GPU usage over time.
    torch.cuda.empty_cache()

    # Write timing (also indicates success) file.
    filepath = os.path.join(config.directories.reports, 'gpu-usage', f'{name}-time.csv')
    timer.save(filepath)

    print('Training complete.')
