from datetime import datetime
from functools import reduce
from GPUtil import getGPUs, showUtilization
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import SGD
from torch.utils.checkpoint import checkpoint
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List, Optional, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.training import exists
from mymi.geometry import get_centre
from mymi.loaders import MultiLoader
from mymi import logging
from mymi.losses import TverskyWithFocalLoss
from mymi.models.networks import MutilUNet3DMemoryTest
from mymi.regions import to_list
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi import types
from mymi.utils import append_row, arg_to_list, save_csv

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_memory_test(
    dataset: Union[str, List[str]],
    model: str,
    run: str) -> None:
    logging.arg_log('Training model', ('dataset', 'model', 'run'), (dataset, model, run))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 'libgcc'
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    # Create model.
    n_channels = 5
    model = MutilUNet3DMemoryTest(n_output_channels=n_channels).to(device)

    # Create loss function, optimiser and gradient scaler.
    loss_fn = TverskyWithFocalLoss()
    optimiser = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = GradScaler()

    # Set CUDNN optimisations.
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True

    # Create dummy data.
    shape = (301, 301, 250)
    input_b = torch.rand(1, 1, *shape).to(device)
    label_b = torch.ones(1, n_channels, *shape, dtype=bool).to(device)
    class_mask_b = torch.ones(1, n_channels, dtype=bool).to(device)
    class_weights_b = torch.ones(1, n_channels, dtype=bool).to(device)
    
    # Zero all parameter gradients.
    optimiser.zero_grad()

    # Perform training step.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, use_cuda=True) as prof:
        with record_function('training-step'):
            with autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                output_b = model(input_b)
                loss = loss_fn(output_b, label_b, class_mask_b, class_weights_b)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

    prof.export_chrome_trace('trace.json')

    print('Training complete.')
