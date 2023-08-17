from datetime import datetime
import json
import numpy as np
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.tuner import Tuner
import torch
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
    complexity_weights_factor: float = 1,
    complexity_weights_window: int = 5,
    dw_cvg_delay_above: int = 20,
    dw_cvg_delay_below: int = 5,
    dw_cvg_thresholds: List[float] = [],
    dw_factor: float = 1,
    halve_channels: bool = False,
    include_background: bool = False,
    lam: float = 0.5,
    loss_fn: str = 'dice_with_focal',
    lr_find: bool = False,
    lr_find_iter: int = 1e3,
    lr_find_min_lr: float = 1e-6,
    lr_find_max_lr: float = 1e3,
    lr_init: float = 1e-3,
    lr_milestones: List[int] = [],
    n_epochs: int = 100,
    n_folds: Optional[int] = None,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_workers: int = 1,
    n_split_channels: int = 2,
    p_val: float = 0.2,
    precision: Union[str, int] = 'bf16',
    random_seed: float = 0,
    resume: bool = False,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    stand_mean: Optional[float] = None,
    stand_std: Optional[float] = None,
    test_fold: Optional[int] = None,
    thresh_low: Optional[float] = None,
    thresh_high: Optional[float] = None,
    use_augmentation: bool = True,
    use_complexity_weights: bool = False,
    use_downweighting: bool = False,
    use_loader_split_file: bool = False,
    use_logger: bool = False,
    use_lr_scheduler: bool = False,
    use_stand: bool = False,
    use_thresh: bool = False,
    use_weights: bool = False,
    weight_decay: float = 0,
    weights: Optional[List[float]] = None,
    weights_iv_factor: Optional[int] = None) -> None:
    logging.arg_log('Training model', ('dataset', 'region', 'model_name', 'run_name'), (dataset, region, model_name, run_name))
    regions = arg_to_list(region, str)

    # Ensure model parameter initialisation is deterministic.
    seed_everything(random_seed, workers=True)

    # Get augmentation transforms.
    if use_augmentation:
        transform_train, transform_val = get_transforms(thresh_high=thresh_high, thresh_low=thresh_low, use_stand=use_stand, use_thresh=use_thresh)
    else:
        transform_train = None
        transform_val = None
    logging.info(f"Training transform: {transform_train}")
    logging.info(f"Validation transform: {transform_val}")

    # Define loss function.
    if loss_fn == 'dice':
        loss_fn = DiceLoss()
    elif loss_fn == 'dice_with_focal':
        loss_fn = DiceWithFocalLoss(lam=lam)

    # Calculate volume weights.
    if use_weights and weights_iv_factor is not None:
        assert weights is None
        inv_volumes = np.array([
            1.81605095e-05,
            3.75567497e-05,
            1.35979999e-04,
            1.34588032e-04,
            1.71684281e-03,
            1.44678695e-03,
            1.63991258e-03,
            3.45656440e-05,
            3.38292316e-05
        ])
        weights = list(inv_volumes ** (1 / weights_iv_factor))

    # Get checkpoint path.
    # Also get training epoch so that our random sampler knows which seed to use
    # when shuffling training data for this epoch.
    opt_kwargs = {}
    if resume:
        # Get checkpoint path.
        if resume_run is not None:
            ckpt_path = os.path.join(config.directories.models, model_name, resume_run, f'{resume_ckpt}.ckpt')
        else:
            ckpt_path = os.path.join(ckpts_path, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path

        # Get training epoch.
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        epoch = ckpt['epoch'] + 1
    else:
        epoch = 0

    # Create data loaders.
    train_loader, val_loader, _ = MultiLoader.build_loaders(dataset, batch_size=batch_size, data_hook=naive_crop, epoch=epoch, include_background=include_background, n_folds=n_folds, n_workers=n_workers, p_val=p_val, random_seed=random_seed, region=regions, test_fold=test_fold, transform_train=transform_train, transform_val=transform_val, use_split_file=use_loader_split_file)

    # Create model.
    model = MultiSegmenter(
        complexity_weights_factor=complexity_weights_factor,
        complexity_weights_window=complexity_weights_window,
        dw_factor=dw_factor,
        dw_cvg_delay_above=dw_cvg_delay_above,
        dw_cvg_delay_below=dw_cvg_delay_below,
        dw_cvg_thresholds=dw_cvg_thresholds,
        loss=loss_fn,
        lr_init=lr_init,
        lr_milestones=lr_milestones,
        halve_channels=halve_channels,
        metrics=['dice'],
        n_gpus=n_gpus,
        n_split_channels=n_split_channels,
        random_seed=random_seed,
        region=regions,
        use_complexity_weights=use_complexity_weights,
        use_downweighting=use_downweighting,
        use_lr_scheduler=use_lr_scheduler,
        use_weights=use_weights,
        weights=weights,
        weight_decay=weight_decay)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            # group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.reports)
        logger.watch(model, log='all') # Caused multi-GPU training to hang.
    else:
        logger = False

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
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    
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
        tuner = Tuner(trainer)
        lr = tuner.lr_find(model, train_loader, val_loader, early_stop_threshold=None, min_lr=lr_find_min_lr, max_lr=lr_find_max_lr, num_training=lr_find_iter)
        logging.info(lr.results)
        filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(json.dumps(lr.results))

        # Don't proceed with training.
        return

    # Save training information.
    man_df = get_multi_loader_manifest(dataset, n_folds=n_folds, region=regions, test_fold=test_fold, use_split_file=use_loader_split_file)
    folderpath = os.path.join(config.directories.runs, model_name, run_name, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'multi-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
