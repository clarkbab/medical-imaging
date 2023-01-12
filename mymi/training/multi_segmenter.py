from datetime import datetime
from monai.losses import DiceFocalLoss
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
from typing import List, Optional, Tuple, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.training import exists
from mymi.loaders import MultiLoader
from mymi import logging
from mymi.losses import TverskyWithFocalLoss
from mymi.models.systems import MultiSegmenter
from mymi.regions import to_list
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi import types
from mymi.utils import arg_log, arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_multi_segmenter(
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
    pretrained_model: Optional[types.ModelName] = None,    
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
    arg_log('Training model', ('dataset', 'model', 'run'), (dataset, model, run))

    # 'libgcc'
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    # Load datasets and check for consistent spacing.
    datasets = arg_to_list(dataset, str)
    spacing_0 = ds.get(datasets[0], 'training').params['spacing']
    for dataset in datasets[1:]:
        spacing = ds.get(dataset, 'training').params['spacing']
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
    train_loader, val_loader, _ = MultiLoader.build_loaders(datasets, n_folds=n_folds, n_train=n_train, n_workers=n_workers, p_val=p_val, regions=regions, spacing=spacing, test_fold=test_fold, transform=transform)

    # Get loss function.
    loss_fn = TverskyWithFocalLoss()

    # Create model.
    metrics = ['dice']
    if pretrained_model:
        pretrained_model = MultiSegmenter.load(*pretrained_model)
    model = MultiSegmenter(
        regions,
        loss=loss_fn,
        metrics=metrics,
        pretrained_model=pretrained_model,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            # group=f"{model_name}-{run}",
            project=model_name,
            name=run,
            save_dir=config.directories.reports)
        logger.watch(model)   # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    checks_path = os.path.join(config.directories.models, model_name, run)
    callbacks = [
        # EarlyStopping(
        #     monitor='val/loss',
        #     patience=5),
        ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=checks_path,
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            monitor='val/loss',
            save_last=True,
            save_top_k=1)
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        # Get the checkpoint path.
        resume_run = resume_run if resume_run is not None else run
        logging.info(f'Loading ckpt {model_name}, {resume_run}, {resume_ckpt}')
        ckpt_path = os.path.join(config.directories.models, model_name, resume_run, f'{resume_ckpt}.ckpt')
        opt_kwargs['ckpt_path'] = ckpt_path

    # Perform training.
    trainer = Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=2,
        precision=16,
        strategy='ddp')

    if lr_find:
        lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
        logging.info(lr_finder.results)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        exit()

    # Save training information.
    man_df = get_multi_loader_manifest(datasets, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
    folderpath = os.path.join(config.directories.runs, model_name, run, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'multi-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
