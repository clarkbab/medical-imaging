from datetime import datetime
from monai.losses import DiceFocalLoss
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
from typing import List, Optional, Union

from mymi import config
from mymi.loaders import MultiLoader
from mymi import logging
from mymi.losses import DiceLoss, DiceWithFocalLoss
from mymi.models.systems import MultiSegmenter
from mymi.reporting.loaders import get_multi_loader_manifest
from mymi.transforms import centre_crop_or_pad_3D
from mymi.utils import arg_to_list

DATETIME_FORMAT = '%Y_%m_%d_%H_%M_%S'

def train_localiser_replan(
    dataset: Union[str, List[str]],
    region: str,
    model_name: str,
    run_name: str,
    halve_channels: bool = False,
    lr_init: float = 1e-3,
    lr_find: bool = False,
    n_epochs: int = 100,
    n_folds: Optional[int] = 5,
    n_gpus: int = 1,
    n_nodes: int = 1,
    n_workers: int = 1,
    p_val: float = 0.2,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    test_fold: Optional[int] = None,
    use_logger: bool = False) -> None:
    logging.arg_log('Training model', ('dataset', 'region', 'model_name', 'run_name'), (dataset, region, model_name, run_name))
    datasets = arg_to_list(dataset, str)

    # Create transforms.
    rotation = (-5, 5)
    translation = (-50, 50)
    scale = (0.8, 1.2)
    transform = RandomAffine(
        degrees=rotation,
        scales=scale,
        translation=translation,
        default_pad_value='minimum')

    # Define loss function.
    loss_fn = DiceWithFocalLoss()
    # loss_fn = DiceLoss()

    # Define crop function.
    crop_x_mm = 300
    def crop_x(input, labels, spacing=None):
        assert spacing is not None

        # Crop input.
        crop = list(input.shape)
        crop[0] = int(crop_x_mm / spacing[0])
        input = centre_crop_or_pad_3D(input, crop)

        # Crop labels.
        for r in labels.keys():
            labels[r] = centre_crop_or_pad_3D(labels[r], crop)

        return input, labels

    # Create data loaders.
    train_loader, val_loader, _ = MultiLoader.build_loaders(datasets, data_hook=None, n_folds=n_folds, n_workers=n_workers, p_val=p_val, region=region, test_fold=test_fold, transform=transform)

    # Create model.
    model = MultiSegmenter(
        region,
        loss_fn,
        lr_init=lr_init,
        halve_channels=halve_channels,
        metrics=['dice'],
        n_gpus=n_gpus)

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
    checks_path = os.path.join(config.directories.models, model_name, run_name)
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
            save_top_k=3),
        LearningRateMonitor(logging_interval='epoch')
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        if resume_checkpoint is None:
            raise ValueError(f"Must pass 'resume_checkpoint' when resuming training run.")
        check_path = os.path.join(checks_path, f"{resume_checkpoint}.ckpt")
        opt_kwargs['resume_from_checkpoint'] = check_path
    
    # Perform training.
    trainer = Trainer(
        accelerator='gpu' if n_gpus > 0 else 'cpu',
        callbacks=callbacks,
        devices=list(range(n_gpus)) if n_gpus > 0 else 1,
        logger=logger,
        max_epochs=n_epochs,
        num_nodes=n_nodes,
        num_sanity_val_steps=2,
        precision=16)

    if lr_find:
        lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader, early_stop_threshold=None, update_attr=True)
        logging.info(lr_finder.results)
        logging.info(lr_finder.suggestion())
        exit()

    # Save training information.
    man_df = get_multi_loader_manifest(datasets, n_folds=n_folds, region=region, test_fold=test_fold)
    folderpath = os.path.join(config.directories.runs, model_name, run_name, datetime.now().strftime(DATETIME_FORMAT))
    os.makedirs(folderpath, exist_ok=True)
    filepath = os.path.join(folderpath, 'multi-loader-manifest.csv')
    man_df.to_csv(filepath, index=False)

    # Train the model.
    trainer.fit(model, train_loader, val_loader, **opt_kwargs)
