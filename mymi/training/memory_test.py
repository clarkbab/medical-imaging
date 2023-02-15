import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import sleep
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import SGD
from typing import Literal, Optional

from mymi import config
from mymi import logging
from mymi.losses import TverskyWithFocalLoss
from mymi.models.networks import MemoryTest
from mymi.reporting.gpu_usage import record_gpu_usage
from mymi.utils import Timer

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_memory_test(
    ckpt_library: Literal['baseline', 'ckpt-pytorch', 'ckpt-fairscale', 'ckpt-offload'],
    ckpt_mode: Literal['', '-level'],
    input_mode: Literal['', '-small', '-xsmall'],
    n_ckpts: int,
    monitor_time: float = 15,
    n_train_steps: int = 11) -> None:
    logging.arg_log('Running memory test', ('ckpt_library', 'ckpt_mode', 'input_mode', 'n_ckpts'), (ckpt_library, ckpt_mode, input_mode, n_ckpts))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Determine input shape.
    if input_mode == '-large':
        shape = (370, 370, 250)
    elif input_mode == '':
        shape = (360, 360, 250)
    elif input_mode == '-small':
        shape = (300, 300, 250)
    elif input_mode == '-xsmall':
        shape = (200, 200, 250)
    else:
        raise ValueError(f"'input_mode={input_mode}' not recognised.")

    # Create name.
    name = f"memory-test{input_mode}-{ckpt_library}-n_ckpts-{n_ckpts}{ckpt_mode}"

    # 'libgcc'
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    # Create model.
    n_channels = 5
    model = MemoryTest(n_channels, ckpt_library=ckpt_library, n_ckpts=n_ckpts).to(device)

    # Create loss function, optimiser and gradient scaler.
    loss_fn = TverskyWithFocalLoss()
    optimiser = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = GradScaler()

    # Set CUDNN optimisations.
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True

    # Kick off GPU memory recording.
    thread = Thread(target=record_gpu_usage, args=(name, monitor_time, 1e-3))
    thread.start()
    sleep(2)

    # Create timer.
    timer = Timer(columns={ 'step': int })

    for step in range(n_train_steps):
        # Create dummy data.
        input_b = torch.rand(1, 1, *shape).to(device)
        label_b = torch.ones(1, n_channels, *shape, dtype=bool).to(device)
        class_mask_b = torch.ones(1, n_channels, dtype=bool).to(device)
        class_weights_b = torch.ones(1, n_channels, dtype=bool).to(device)
        
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

    # Write timing file.
    filepath = os.path.join(config.directories.reports, 'gpu-usage', f'{name}-time.csv')
    timer.save(filepath)

    print('Training complete.')
