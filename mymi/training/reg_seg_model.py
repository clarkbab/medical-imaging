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
from mymi.loaders import RegSegLoader
from mymi.loaders.augmentation import get_transforms
from mymi import logging
from mymi.losses import NCCLoss, DiceWithFocalLoss
from mymi.models import replace_ckpt_alias
from mymi.models.systems import RegSegModel
from mymi.regions import RegionList, region_to_list
from mymi.reporting.loaders import get_reg_seg_loader_manifest
from mymi.types import PatientRegions
from mymi.utils import arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_reg_seg_model(
    dataset: Union[str, List[str]],
    region: PatientRegions,
    model_name: str,
    run_name: str,
    batch_size: int = 1,
    ckpt_model: bool = True,
    complexity_weights_factor: float = 1,
    complexity_weights_window: int = 5,
    cw_cvg_calculate: bool = True,
    cw_cvg_delay_above: int = 20,
    cw_cvg_delay_below: int = 5,
    cw_factor: Optional[Union[float, List[float]]] = None,
    cw_schedule: Optional[List[float]] = None,
    cyclic_min: Optional[float] = None,
    cyclic_max: Optional[float] = None,
    dilate_iters: Optional[List[int]] = None,
    dilate_region: Optional[PatientRegions] = None,
    dilate_schedule: Optional[List[int]] = None,
    grad_acc: int = 1,
    halve_channels: bool = False,
    lam: float = 0.5,
    loader_load_all_samples: bool = False,
    loader_shuffle_samples: bool = True,
    loss_fn: str = 'ncc',
    lr_find: bool = False,
    lr_find_min_lr: float = 1e-6,
    lr_find_max_lr: float = 1,
    lr_find_n_iter: int = 1000,
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
    resume_model: Optional[str] = None,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    resume_ckpt_version: Optional[int] = None,
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
    use_cvg_weighting: bool = False,
    use_dilation: bool = False,
    use_elastic: bool = False,
    use_loader_grouping: bool = False,
    use_loader_split_file: bool = False,
    use_logger: bool = False,
    use_lr_scheduler: bool = False,
    use_stand: bool = False,
    use_thresh: bool = False,
    use_weights: bool = False,
    val_image_interval: int = 50,
    weight_decay: float = 0,
    weights: Optional[List[float]] = None,
    weights_iv_factor: Optional[Union[float, List[float]]] = None,
    weights_schedule: Optional[List[float]] = None) -> None:
    logging.arg_log('Training model', ('dataset', 'model_name', 'run_name'), (dataset, model_name, run_name))
    regions = region_to_list(region)

    # Ensure model parameter initialisation is deterministic.
    seed_everything(random_seed, workers=True)

    # Get augmentation transforms.
    if use_augmentation:
        transform_train, transform_val = get_transforms(thresh_high=thresh_high, thresh_low=thresh_low, use_elastic=use_elastic, use_stand=use_stand, use_thresh=use_thresh)
    else:
        transform_train = None
        transform_val = None
    logging.info(f"Training transform: {transform_train}")
    logging.info(f"Validation transform: {transform_val}")

    # Define loss function.
    if loss_fn == 'ncc':
        loss_fn = NCCLoss()
    elif loss_fn == 'dice_with_focal':
        loss_fn = DiceWithFocalLoss(lam=lam)
    logging.info(f"Using loss function: {loss_fn}")

    # Calculate volume weights.
    if use_weights and weights_iv_factor is not None:
        if weights is not None:
            raise ValueError(f"Can't specify both 'weights' and 'weights_iv_factor'.")

        weights_iv_factors = arg_to_list(weights_iv_factor, float)
        if weights_schedule is not None:
            if len(weights_iv_factors) != len(weights_schedule):
                raise ValueError(f"When using inverse volume weighting, number of factors ({weights_iv_factors}, len={len(weights_iv_factors)}) should equal number of schedule points ({weights_schedule}, len={len(weights_schedule)}).")
        else:
            if len(weights_iv_factors) != 1:
                raise ValueError(f"When using inverse volume weighting with multiple factors ({len(weights_iv_factors)}), must pass 'weights_schedule'.")

        # Infer inverse volumes from dataset name.
        first_dataset = arg_to_list(dataset, str)[0]
        if 'MICCAI-2015' in first_dataset:
            all_regions = RegionList.MICCAI
            all_inv_volumes = RegionList.MICCAI_INVERSE_VOLUMES
            region_idxs = [all_regions.index(r) for r in regions]
            inv_volumes = [all_inv_volumes[i] for i in region_idxs]
        elif 'PMCC-HN-REPLAN' in first_dataset:
            inv_volumes = RegionList.PMCC_REPLAN_INVERSE_VOLUMES
        elif 'PMCC' in first_dataset:
            all_regions = RegionList.PMCC
            all_inv_volumes = RegionList.PMCC_INVERSE_VOLUMES
            region_idxs = [all_regions.index(r) for r in regions]
            inv_volumes = [all_inv_volumes[i] for i in region_idxs]
        else:
            raise ValueError(f"Couldn't infer 'inv_volumes' from dataset name '{first_dataset}'.")

        # Calculate weights.
        weights = []
        schedule = []
        for i, weights_iv_factor in enumerate(weights_iv_factors):
            weights_i = list(np.array(inv_volumes) ** weights_iv_factor)
            weights.append(weights_i)
            schedule_i = weights_schedule[i] if weights_schedule is not None else 0
            schedule.append(schedule_i)
        weights_schedule = schedule

    # Get checkpoint path.
    # Also get training epoch so that our random sampler knows which seed to use
    # when shuffling training data for this epoch.
    opt_kwargs = {}
    if resume:
        # Get checkpoint path.
        resume_model = resume_model if resume_model is not None else model_name
        resume_run = resume_run if resume_run is not None else run_name
        _, _, resume_ckpt = replace_ckpt_alias((resume_model, resume_run, resume_ckpt), ckpt_version=resume_ckpt_version)
        ckpt_path = os.path.join(config.directories.models, resume_model, resume_run, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path

        # Get training epoch.
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        epoch = ckpt['epoch'] + 1
    else:
        epoch = 0

    # Create data loaders.
    train_loader, val_loader, _ = RegSegLoader.build_loaders(dataset, batch_size=batch_size, epoch=epoch, load_all_samples=loader_load_all_samples, n_folds=n_folds, n_workers=n_workers, p_val=p_val, random_seed=random_seed, region=regions, shuffle_samples=loader_shuffle_samples, test_fold=test_fold, transform_train=transform_train, transform_val=transform_val, use_grouping=use_loader_grouping, use_split_file=use_loader_split_file)

    # Infer convergence thresholds from dataset name.
    # We need these even when 'use_cvg_weighting=False' as it allows us to track
    # converged OARs via wandb API.
    if cw_cvg_calculate:
        first_dataset = arg_to_list(dataset, str)[0]
        if 'MICCAI-2015' in first_dataset:
            all_regions = RegionList.MICCAI
            all_thresholds = RegionList.MICCAI_CVG_THRESHOLDS
            region_idxs = [all_regions.index(r) for r in regions]
            cw_cvg_thresholds = [all_thresholds[i] for i in region_idxs]
        elif 'PMCC-HN-REPLAN' in first_dataset:
            cw_cvg_thresholds = RegionList.PMCC_REPLAN_CVG_THRESHOLDS
        elif 'PMCC' in first_dataset:
            all_regions = RegionList.PMCC
            all_thresholds = RegionList.PMCC_CVG_THRESHOLDS
            region_idxs = [all_regions.index(r) for r in regions]
            cw_cvg_thresholds = [all_thresholds[i] for i in region_idxs]
        else:
            raise ValueError(f"Couldn't infer 'cw_cvg_thresholds' from dataset name '{first_dataset}'.")
    else:
        cw_cvg_thresholds = None

    # Create model.
    model = RegSegModel(
        complexity_weights_factor=complexity_weights_factor,
        complexity_weights_window=complexity_weights_window,
        cw_cvg_calculate=cw_cvg_calculate,
        cw_cvg_delay_above=cw_cvg_delay_above,
        cw_cvg_delay_below=cw_cvg_delay_below,
        cw_cvg_thresholds=cw_cvg_thresholds,
        cw_factor=cw_factor,
        cw_schedule=cw_schedule,
        cyclic_min=cyclic_min,
        cyclic_max=cyclic_max,
        dilate_iters=dilate_iters,
        dilate_region=dilate_region,
        dilate_schedule=dilate_schedule,
        loss=loss_fn,
        lr_find=lr_find,
        lr_init=lr_init,
        lr_milestones=lr_milestones,
        halve_channels=halve_channels,
        metrics=['ncc'],
        model_name=model_name,
        n_gpus=n_gpus,
        n_split_channels=n_split_channels,
        random_seed=random_seed,
        region=regions,
        run_name=run_name,
        use_complexity_weights=use_complexity_weights,
        use_cvg_weighting=use_cvg_weighting,
        use_dilation=use_dilation,
        use_lr_scheduler=use_lr_scheduler,
        use_weights=use_weights,
        val_image_interval=val_image_interval,
        weights=weights,
        weights_schedule=weights_schedule,
        weight_decay=weight_decay)

    # Create logger.
    if use_logger:
        logging.info(f"Creating Wandb logger.")

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
        accumulate_grad_batches=grad_acc,
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=0,
        precision=precision)

    if lr_find:
        logging.arg_log('Performing LR find', ('min_lr', 'max_lr', 'n_iter'), (lr_find_min_lr, lr_find_max_lr, lr_find_n_iter))
        tuner = Tuner(trainer)
        lr = tuner.lr_find(model, train_loader, val_loader, early_stop_threshold=None, min_lr=lr_find_min_lr, max_lr=lr_find_max_lr, num_training=lr_find_n_iter)
        logging.info(lr.results)
        filepath = os.path.join(config.directories.models, model_name, run_name, 'lr-finder.json')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(json.dumps(lr.results))

        # Don't proceed with training.
        return

    # Save training information.
    man_df = get_reg_seg_loader_manifest(dataset, load_all_samples=loader_load_all_samples, n_folds=n_folds, shuffle_samples=loader_shuffle_samples, test_fold=test_fold, use_grouping=use_loader_grouping, use_split_file=use_loader_split_file)
    folderpath = os.path.join(config.directories.runs, model_name, run_name, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'reg-seg-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
