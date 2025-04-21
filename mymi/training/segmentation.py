import os
import pytorch_lightning as pl
from typing import *

from mymi import config
from mymi.datasets import TrainingDataset
from mymi.loaders import HoldoutLoader
from mymi.loaders.augmentation import RandomAffine, get_transforms
from mymi import logging
from mymi.models import replace_ckpt_alias
from mymi.models.lightning_modules import *
from mymi.regions import regions_to_list
from mymi.typing import *

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_segmenter(
    dataset: str,
    regions: Regions,
    model_name: str,
    run_name: str,
    arch: str = 'unet3d:M',
    batch_size: int = 1,
    ckpt_model: bool = True,
    gn_alpha: float = 1.0,
    gn_balance_point_type: Literal['output', 'params'] = 'params',
    gn_clip_mult: Optional[float] = None,
    gn_enabled: bool = True,
    gn_loss_fn: Literal['abs', 'square'] = 'abs',
    gn_lr_init: float = 1e-3,
    gn_norm_limit_mult: Optional[float] = None,
    gn_softmax: bool = False,
    loss_fn: str = 'dice',
    loss_smoothing: float = 0,
    lr_init: float = 1e-3,
    ls_func: Literal['abs', 'square'] = 'square',
    model_type: Literal['Segmenter', 'SegmenterGradNorm', 'SegmenterTversky'] = 'Segmenter',
    n_epochs: int = 1000,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_val: Optional[int] = None,
    n_workers: int = 1,
    pc_reduction: Literal['mean', 'sum'] = 'mean',
    precision: Union[str, int] = '32',
    random_seed: float = 42,
    resume: bool = False,
    resume_model: Optional[str] = None,
    resume_run: Optional[str] = None,
    resume_ckpt: str = 'last',
    resume_ckpt_version: Optional[int] = None,
    save_training_metrics: bool = False,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    use_augmentation: bool = True,
    use_wandb: bool = True,
    **kwargs) -> None:
    logging.arg_log('Training model', ('dataset', 'regions', 'model_name', 'run_name'), (dataset, regions, model_name, run_name))
    regions = regions_to_list(regions, literals={ 'all': TrainingDataset(dataset).regions })

    # Ensure model parameter initialisation is deterministic.
    pl.seed_everything(random_seed, workers=True)

    # Get checkpoint path.
    opt_kwargs = {}
    if resume:
        # Get checkpoint path.
        resume_model = resume_model if resume_model is not None else model_name
        resume_run = resume_run if resume_run is not None else run_name
        resume_ckpt = replace_ckpt_alias(resume_model, resume_run, resume_ckpt, ckpt_version=resume_ckpt_version)
        ckpt_path = os.path.join(config.directories.models, resume_model, resume_run, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path

    # Get augmentation transforms.
    if use_augmentation:
        # transform_train, transform_val = get_transforms()
        transform_train = RandomAffine()
        transform_val = None
    else:
        transform_train = None
        transform_val = None
    logging.info(f"Training transform: {transform_train}")
    logging.info(f"Validation transform: {transform_val}")

    # Create data loaders.
    okwargs = dict(
        batch_size=batch_size,
        model=f'{model_name}/{run_name}',
        n_val=n_val,
        n_workers=n_workers,
        regions=regions,
        transform_train=transform_train,
    )
    tl, vl, _, _ = HoldoutLoader.build_loaders(dataset, **okwargs)

    # Create model.
    okwargs = dict(
        arch=arch,
        gn_alpha=gn_alpha,
        gn_balance_point_type=gn_balance_point_type,
        gn_clip_mult=gn_clip_mult,
        gn_enabled=gn_enabled,
        gn_loss_fn=gn_loss_fn,
        gn_lr_init=gn_lr_init,
        gn_norm_limit_mult=gn_norm_limit_mult,
        gn_softmax=gn_softmax,
        loss_fn=loss_fn,
        loss_smoothing=loss_smoothing,
        lr_init=lr_init,
        ls_func=ls_func,
        name=(model_name, run_name),
        regions=regions,
        save_training_metrics=save_training_metrics,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
    )
    if model_type == 'Segmenter':
        model = Segmenter(**okwargs)
    elif model_type == 'SegmenterGradNorm':
        model = SegmenterGradNorm(**okwargs)
    elif model_type == 'SegmenterPCGrad':
        model = SegmenterPCGrad(pc_reduction=pc_reduction, **okwargs)
    elif model_type == 'SegmenterTversky':
        model = SegmenterTversky(**okwargs)
    elif model_type == 'SegmenterUncertaintyWeighting':
        model = SegmenterUncertaintyWeighting(**okwargs)
    else:
        raise ValueError(f"Model type '{model_type}' not recognised.")

    # Create logger.
    if use_wandb:
        logging.info(f"Creating Wandb logger.")

        logger = pl.loggers.WandbLogger(
            project=model_name,
            name=run_name,
            save_dir=config.directories.reports)
        # logger.watch(model, log='all') # Caused multi-GPU training to hang.
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
        callbacks.append(pl.callbacks.ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=ckpts_path,
            # filename='dice={val/dice:.6f}-epoch={epoch}-step={trainer/global_step}',
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            # mode='max',
            mode='min',
            # monitor='val/dice',
            monitor='val/loss',
            save_last=True,
            save_top_k=1))
    if logger:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    
    # Perform training.
    trainer = pl.Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=1,
        precision=precision)

    # Train the model.
    trainer.fit(model, tl, vl, **opt_kwargs)
