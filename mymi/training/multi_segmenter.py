from datetime import datetime
import json
import numpy as np
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from typing import List, Optional, Union

from mymi import config
from mymi.loaders import MultiLoader
from mymi.loaders.augmentation import get_transforms
from mymi.loaders.hooks import naive_crop
from mymi import logging
from mymi.losses import DiceLoss, DiceWithFocalLoss
from mymi.models.systems import MultiSegmenter, Segmenter
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi.utils import arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_multi_segmenter(
    dataset: Union[str, List[str]],
    region: str,
    model_name: str,
    run_name: str,
    batch_size: int = 1,
    ckpt_model: bool = True,
    halve_channels: bool = False,
    lam: float = 0.5,
    loss_fn: str = 'dice_with_focal',
    lr_find: bool = False,
    lr_find_min_lr: float = 1e-6,
    lr_find_max_lr: float = 1e3,
    lr_find_num_train: int = 1e3,
    lr_init: float = 1e-3,
    n_epochs: int = 100,
    n_folds: Optional[int] = None,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_workers: int = 1,
    n_split_channels: int = 1,
    p_val: float = 0.2,
    precision: Union[str, int] = 16,
    resume: bool = False,
    resume_ckpt: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    stand_mean: Optional[float] = None,
    stand_std: Optional[float] = None,
    test_fold: Optional[int] = None,
    thresh_low: Optional[float] = None,
    thresh_high: Optional[float] = None,
    use_loader_split_file: bool = False,
    use_logger: bool = False,
    use_lr_scheduler: bool = False,
    use_stand: bool = False,
    use_thresh: bool = False,
    weight_decay: float = 0) -> None:
    logging.arg_log('Training model', ('dataset', 'region', 'model_name', 'run_name'), (dataset, region, model_name, run_name))
    datasets = arg_to_list(dataset, str)

    # Get augmentation transforms.
    transform_train, transform_val = get_transforms(thresh_high=thresh_high, thresh_low=thresh_low, use_stand=use_stand, use_thresh=use_thresh)
    logging.info(f"Training transform: {transform_train}")
    logging.info(f"Validation transform: {transform_val}")

    # Define loss function.
    if loss_fn == 'dice':
        loss_fn = DiceLoss()
    elif loss_fn == 'dice_with_focal':
        loss_fn = DiceWithFocalLoss(lam=lam)

    # Create data loaders.
    train_loader, val_loader, _ = MultiLoader.build_loaders(datasets, batch_size=batch_size, data_hook=naive_crop, n_folds=n_folds, n_workers=n_workers, p_val=p_val, region=region, test_fold=test_fold, transform_train=transform_train, transform_val=transform_val, use_split_file=use_loader_split_file)

    # Create model.
    model = MultiSegmenter(
        loss=loss_fn,
        lr_init=lr_init,
        halve_channels=halve_channels,
        metrics=['dice'],
        n_gpus=n_gpus,
        n_split_channels=n_split_channels,
        region=region,
        use_lr_scheduler=use_lr_scheduler,
        weight_decay=weight_decay)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            # group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.reports)
        logger.watch(model) # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    ckpts_path = os.path.join(config.directories.models, model_name, run_name)
    callbacks = [
        # EarlyStopping(
        #     monitor='val/loss',
        #     patience=5),
    ]
    if ckpt_model:
        callbacks.append(ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=ckpts_path,
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            monitor='val/loss',
            save_last=True,
            save_top_k=1))
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        if resume_ckpt is None:
            raise ValueError(f"Must pass 'resume_ckpt' when resuming training run.")
        opt_kwargs['ckpt_path'] = os.path.join(ckpts_path, f'{resume_ckpt}.ckpt')
    
    # Perform training.
    trainer = Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=2,
        precision=precision)

    if lr_find:
        lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader, early_stop_threshold=None, min_lr=lr_find_min_lr, max_lr=lr_find_max_lr, num_training=lr_find_num_train)
        logging.info(lr_finder.results)
        filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(json.dumps(lr_finder.results))
        exit()

    # Save training information.
    man_df = get_multi_loader_manifest(datasets, n_folds=n_folds, region=region, test_fold=test_fold, use_split_file=use_loader_split_file)
    folderpath = os.path.join(config.directories.runs, model_name, run_name, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'multi-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
