from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchio.transforms import RandomAffine
from typing import List, Optional, Tuple, Union

from mymi import config
from mymi import dataset as ds
from mymi.dataset.training import exists
from mymi.loaders import Loader, MultiLoader
from mymi.losses import DiceLoss, DiceWithFocalLoss
from mymi import logging
from mymi.models.systems import Localiser, MultiSegmenter
from mymi.reporting.loaders import get_loader_manifest
from mymi.utils import arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_localiser(
    dataset: Union[str, List[str]],
    model_name: str,
    run_name: str,
    region: str,
    loss: str = 'dice',
    n_epochs: int = 200,
    n_folds: Optional[int] = 5,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_train: Optional[int] = None,
    n_workers: int = 1,
    precision: Union[int, str] = 16,
    pretrained: Optional[Tuple[str, str, str]] = None,
    p_val: float = 0.2,
    resume: bool = False,
    resume_ckpt: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
    logging.info(f"Training model '({model_name}, {run_name})' on dataset '{dataset}' with region '{region}' using '{n_folds}' folds with test fold '{test_fold}'.")

    # Load datasets.
    datasets = arg_to_list(dataset, str)
    spacing = ds.get(datasets[0], 'training').params['output-spacing']
    for dataset in datasets[1:]:
        # Check for consistent spacing.
        new_spacing = ds.get(dataset, 'training').params['output-spacing']
        if new_spacing != spacing:
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
    train_loader, val_loader, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, n_train=n_train, n_workers=n_workers, p_val=p_val, spacing=spacing, test_fold=test_fold, transform=transform)

    # Get loss function.
    if loss == 'dice':
        # loss_fn = DiceLoss()
        loss_fn = DiceWithFocalLoss()
    elif loss == 'scdice':
        loss_fn = DiceLoss(weights=[0, 1])

    # Create model.
    metrics = ['dice', 'hausdorff', 'surface']
    if pretrained:
        pretrained = Localiser.load(*pretrained)
    model = Localiser(
        loss=loss_fn,
        metrics=['dice'],
        pretrained=pretrained,
        spacing=spacing)

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
        ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=ckpts_path,
            filename='loss={val/loss:.6f}-epoch={epoch}-step={trainer/global_step}',
            every_n_epochs=1,
            monitor='val/loss',
            save_last=True,
            save_top_k=1)
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        if resume_ckpt is None:
            raise ValueError(f"Must pass 'resume_ckpt' when resuming training run.")
        ckpt_path = os.path.join(ckpts_path, f"{resume_ckpt}.ckpt")
        opt_kwargs['ckpt_path'] = ckpt_path
    
    # Perform training.
    trainer = Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=0,
        precision=precision)

    # Save training information.
    man_df = get_loader_manifest(datasets, region, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
    folderpath = os.path.join(config.directories.runs, model_name, run_name, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
