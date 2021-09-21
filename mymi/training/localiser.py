import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
from typing import List, Optional, Union

from mymi import config
from mymi import dataset as ds
from mymi.loaders import Loader
from mymi import logging
from mymi.models.systems import Localiser
from mymi import types

def train_localiser(
    model_name: str,
    run_name: str,
    datasets: Union[str, List[str]],
    num_epochs: int = 200,
    num_gpus: int = 1,
    num_nodes: int = 1,
    num_subset: Optional[int] = None,
    num_workers: int = 1,
    regions: types.PatientRegions = 'all',
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    use_logger: bool = False) -> None:
    logging.info(f"Training model '({model_name}, {run_name})' on datasets '{datasets}' with regions '{regions}'.")

    # Load partitions.
    if isinstance(datasets, str):
        set = ds.get(datasets, 'processed')
        spacing = eval(set.params().spacing[0])
        train_parts = set.partition('train')
        val_parts = set.partition('validation')
    else:
        set = ds.get(datasets[0], 'processed')
        spacing = eval(set.params().spacing[0]) 
        train_parts = []
        val_parts = []
        for d in datasets:
            set = ds.get(d, 'processed')
            d_spacing = eval(set.params().spacing[0]) 
            if d_spacing != spacing:
                raise ValueError(f"Can't train on datasets with inconsistent spacing.")
            train_parts.append(set.partition('train'))
            val_parts.append(set.partition('validation'))

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
    if num_subset is not None:
        train_loader = SubsetLoader.build(train_parts, num_subset, num_workers=num_workers, regions=regions, spacing=spacing, transform=transform)
    else:
        train_loader = Loader.build(train_parts, num_workers=num_workers, regions=regions, spacing=spacing, transform=transform)
    val_loader = Loader.build(val_parts, num_workers=num_workers, regions=regions, shuffle=False)

    # Create model.
    metrics = ['dice', 'hausdorff']
    model = Localiser(
        region=regions,
        metrics=metrics,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.wandb)
        # logger.watch(model) # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    checks_path = os.path.join(config.directories.checkpoints, model_name, run_name)
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
            save_top_k=3)
    ]

    # Add optional trainer args.
    opt_kwargs = {}
    if resume:
        if resume_checkpoint is None:
            raise ValueError(f"Must pass 'resume_checkpoint' when resuming training run.")
        check_path = os.path.join(checks_path, resume_checkpoint)
        opt_kwargs['resume_from_checkpoint'] = check_path
    
    # Perform training.
    trainer = Trainer(
        accelerator='ddp',
        callbacks=callbacks,
        gpus=list(range(num_gpus)),
        logger=logger,
        max_epochs=num_epochs,
        num_nodes=num_nodes,
        num_sanity_val_steps=0,
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=16,
        **opt_kwargs)
    trainer.fit(model, train_loader, val_loader)
