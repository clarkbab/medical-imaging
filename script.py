from datetime import datetime
import numpy as np
import os
import pandas as pd
from pynvml.smi import nvidia_smi
from threading import Thread
from time import sleep
import torch
from torch import nn

def record_gpu_usage(
    name: str,
    time: float,
    interval: float) -> None:

    # Create results table.
    nvsmi = nvidia_smi.getInstance()
    n_gpus = len(nvsmi.DeviceQuery('gpu_name')['gpu'])
    cols = {
        'time': str
    }
    for i in range(n_gpus):
        device_name = f'cuda:{i}'
        cols[f'{device_name}-usage'] = float
    df = pd.DataFrame(columns=cols.keys())

    # Add usage.
    n_intervals = int(np.ceil(time / interval))
    start_time = datetime.now()
    for i in range(n_intervals):
        # Record GPU usage.
        data = {
            'time': (datetime.now() - start_time).total_seconds()
        }
        usages_mb = [g['fb_memory_usage']['used'] for g in nvsmi.DeviceQuery('memory.used')['gpu']]
        for j, usage_mb in enumerate(usages_mb):
            device_name = f'cuda:{j}'
            data[f'{device_name}-usage'] = usage_mb
        df = pd.concat((df, pd.DataFrame([data])), axis=0)

        # Wait for time interval to pass.
        time_passed = (datetime.now() - start_time).total_seconds()
        if time_passed > time:
            break
        time_to_wait = ((i + 1) * interval) - time_passed
        if time_to_wait > 0:
            sleep(time_to_wait)
        elif time_to_wait < 0:
            # Makes time problem worse if we log.
            # logging.warning(f"GPU usage recording took longer than allocated interval '{interval}' (seconds).")
            pass

    # Save results.
    filepath = f'{name}.csv'
    df.to_csv(filepath, index=False)

n_channels = 64
name = f'testing-{n_channels}'
input_shape = (1, n_channels, 370, 370, 250)
device = torch.device('cuda:0')
input = torch.rand(input_shape)

# Kick off GPU memory recording.
thread = Thread(target=record_gpu_usage, args=(name, 10, 1e-3))
thread.start()
# sleep(3)

input = input.half().to(device)
input_GB = input.numel() * input.element_size() / 1e9
print('input GB: ', input_GB)
input.requires_grad = False
layer = nn.Conv3d(in_channels=n_channels, out_channels=32, kernel_size=3, stride=1, padding=1).half().to(device)
for i, param in enumerate(layer.parameters()):
    param_GB = param.numel() * param.element_size() / 1e9
    print(f'param_{i} GB: {param_GB}')
    param.requires_grad = False
output = layer(input)

torch.cuda.empty_cache()
