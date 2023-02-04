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
from mymi.models.networks import MultiUNet3DCKPT
from mymi.regions import to_list
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi import types
from mymi.utils import append_row, arg_to_list, save_csv

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_memory_test(
    dataset: Union[str, List[str]],
    model: str,
    run: str,
    lr_find: bool = False,
    n_epochs: int = 150,
    n_folds: Optional[int] = 5,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    p_val: float = 0.2,
    regions: types.PatientRegions = 'all',
    resume: bool = False,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
    model_name = model
    logging.arg_log('Training model', ('dataset', 'model', 'run'), (dataset, model, run))

    # 'libgcc'
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    # Create model.
    n_channels = 5
    model = MultiUNet3DCKPT(n_output_channels=n_channels, n_gpus=n_gpus).to('cuda:0')

    # Create loss function, optimiser and gradient scaler.
    loss_fn = TverskyWithFocalLoss()
    optimiser = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = GradScaler()

    # Set CUDNN optimisations.
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True

    # Create dummy data.
    shape = (301, 301, 250)
    input_b = torch.rand(1, 1, *shape).to('cuda:0')
    label_b = torch.ones(1, n_channels, *shape, dtype=bool).to('cuda:0')
    class_mask_b = torch.ones(1, n_channels, dtype=bool).to('cuda:0')
    class_weights_b = torch.ones(1, n_channels, dtype=bool).to('cuda:0')
    
    # Zero all parameter gradients.
    optimiser.zero_grad()

    # Perform forward, backward and update steps.
    with autocast(device_type='cuda', dtype=torch.float16):
        output_b = model(input_b)
        loss = loss_fn(output_b, label_b, class_mask_b, class_weights_b)

    scaler.scale(loss).backward()
    scaler.step(optimiser)
    scaler.update()

    print('Training complete.')
