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
from torchio.transforms import RandomAffine
import torch
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim import SGD
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.training import exists
from mymi.geometry import get_centre
from mymi.loaders import MultiLoader
from mymi import logging
from mymi.losses import TverskyWithFocalLoss
from mymi.models.networks import MultiUNet3D
from mymi.regions import regions_to_list
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi import types
from mymi.utils import append_row, arg_to_list, save_csv

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_multi_segmenter_pytorch(
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

    # Load datasets and check for consistent spacing.
    datasets = arg_to_list(dataset, str)
    spacing_0 = ds.get(datasets[0], 'training').params['output-spacing']
    for d in datasets[1:]:
        spacing = ds.get(d, 'training').params['output-spacing']
        if spacing != spacing_0:
            raise ValueError(f'Datasets must have consistent spacing.')

    # Create transforms.
    rotation = (-5, 5)
    translation = (-50, 50)
    scale = (0.8, 1.2)
    transform = RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')

    # Create data loaders.
    transform = None
    train_loader, val_loader, _ = MultiLoader.build_loaders(datasets, half_precision=False, n_folds=n_folds, n_train=n_train, n_workers=n_workers, p_val=p_val, regions=regions, spacing=spacing, test_fold=test_fold, transform=transform)

    # Save training information.
    man_df = get_multi_loader_manifest(datasets, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
    folderpath = os.path.join(config.directories.runs, model_name, run, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'multi-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Create model.
    n_channels = len(to_list(regions)) + 1
    model = MultiUNet3D(n_output_channels=n_channels, n_gpus=n_gpus)

    # Create loss function, optimiser and gradient scaler.
    loss_fn = TverskyWithFocalLoss()
    optimiser = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scaler = GradScaler()

    # Set CUDNN optimisations.
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Store memory usage.
    cols = {
        'epoch': int,
        'iteration': int,
        'patient-id': str,
        'ct-size': str,
        'num-voxels': int,
        'gpu-mem': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for epoch in range(n_epochs):
        mean_loss = 0

        # Complete training.
        model.train()
        for i, data_b in enumerate(train_loader):
            desc_b, input_b, label_b, class_mask_b, class_weights_b = data_b
            print(f'=== {desc_b[0]} ===')
            print(input_b[0].shape)

            # Crop input and label.
            crop = False
            if crop:
                crop_mm = 400
                crop = int(np.floor(crop_mm / spacing_0[0]))
                x_centre = get_centre(input_b)[2]
                x_lower = x_centre - int(np.floor(crop / 2))
                input_b = input_b[:, :, x_lower:]
                input_b = input_b[:, :, :crop]
                label_b = label_b[:, :, x_lower:]
                label_b = label_b[:, :, :crop]

            # Move loss calculation tensors to final GPU.
            final_gpu = f'cuda:{n_gpus - 1}'
            label_b = label_b.to(final_gpu)
            class_mask_b = class_mask_b.to(final_gpu)
            class_weights_b = class_weights_b.to(final_gpu)
            
            # Zero all parameter gradients.
            optimiser.zero_grad()

            # Perform forward, backward and update steps.
            with autocast(device_type='cuda', dtype=torch.float16):
                dummy_arg = torch.Tensor()
                dummy_arg.requires_grad = True
                # output_b = checkpoint(model, input_b, dummy_arg)
                output_b = model(input_b)
                loss = loss_fn(output_b, label_b, class_mask_b, class_weights_b)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

            # Store GPU usage. Memory should peak after loss is calculate (?)
            # as without checkpointing all intermediate activations are stored.
            print([g.memoryUtil for g in getGPUs()])
            data = {
                'epoch': epoch,
                'iteration': i,
                'patient-id': desc_b[0].split(':')[-1],
                'ct-size': input_b.shape,
                'num-voxels': reduce(np.multiply, input_b.shape),
                'gpu-mem': [g.memoryUtil for g in getGPUs()]
            }
            df = append_row(df, data)

            # Show GPU usage.
            # gpu_interval = 1
            # if i % gpu_interval == gpu_interval - 1:
            #     showUtilization()

            # Print statistics.
            mean_loss += loss.item()
            print_interval = 10
            if i % print_interval == print_interval - 1:
                print(f"[Epoch: {epoch}, Step: {i}] loss: {mean_loss / print_interval:.3f}")
                mean_loss = 0

            # Clear out old tensors.
            del input_b, label_b, class_mask_b, class_weights_b, loss
            torch.cuda.empty_cache()

        # Complete validation.
        model.eval()
        epoch_loss = 0
        for i, data_b in enumerate(val_loader):
            desc_b, input_b, label_b, class_mask_b, class_weights_b = data_b

            # Move loss calculation tensors to final GPU.
            final_gpu = f'cuda:{n_gpus - 1}'
            label_b = label_b.to(final_gpu)
            class_mask_b = class_mask_b.to(final_gpu)
            class_weights_b = class_weights_b.to(final_gpu)

            # Perform forward, backward and update steps.
            with autocast(device_type='cuda', dtype=torch.float16):
                output_b = model(input_b)
                loss = loss_fn(output_b, label_b, class_mask_b, class_weights_b)
            epoch_loss += loss.item()

            # Print statistics.
            mean_loss += loss.item()
            print_interval = 10
            if i % print_interval == print_interval - 1:
                print(f"[Epoch: {epoch}, Validation Step: {i}] loss: {mean_loss / print_interval:.3f}")
                mean_loss = 0

            # Clear out old tensors.
            del input_b, label_b, class_mask_b, class_weights_b, loss
            torch.cuda.empty_cache()

        # Checkpoint model.
        epoch_loss = epoch_loss / len(val_loader)
        

    # Save CSV.
    save_csv(df, 'patient-specific-models', 'data', f'newest-gpu-usage-{n_gpus}-gpus.csv', overwrite=True)

    print('Training complete.')
