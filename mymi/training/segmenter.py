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
from mymi.loaders import PatchLoader
from mymi import logging
from mymi.models.systems import Segmenter
from mymi import types

def train_segmenter(
    model_name: str,
    run_name: str,
    datasets: Union[str, List[str]],
    region: str,
    num_epochs: int = 200,
    num_gpus: int = 1,
    num_nodes: int = 1,
    num_workers: int = 1,
    slurm_job_id: Optional[str] = None,
    slurm_array_task_id: Optional[str] = None,
    use_logger: bool = False) -> None:
    logging.info(f"Training model '({model_name}, {run_name})' on datasets '{datasets}' with region '{region}'.")

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
    patch_size = (128, 128, 96)
    spacing = eval(set.params().spacing[0])
    train_loader = PatchLoader.build(train_parts, patch_size, region, num_workers=num_workers, spacing=spacing, transform=transform)
    val_loader = PatchLoader.build(val_parts, patch_size, region, num_workers=num_workers, shuffle=False)

    # Create map from validation batch_idx to "dataset:partition:sample_idx".
    index_map = dict([(batch_idx, f"{val_parts[part_idx].dataset.name}:validation:{sample_idx}") for batch_idx, (part_idx, sample_idx) in val_loader.dataset._index_map.items()])

    # Create model.
    metrics = ['dice', 'hausdorff', 'surface']
    model = Segmenter(
        index_map=index_map,
        metrics=metrics,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            group=f"{model_name}-{run_name}",
            project=model_name,
            name=run_name,
            save_dir=config.directories.wandb)
        # logger.watch(model)   # Caused multi-GPU training to hang.
    else:
        logger = None

    # Create callbacks.
    path = os.path.join(config.directories.checkpoints, model_name, run_name)
    callbacks = [
        # EarlyStopping(
        #     monitor='val/loss',
        #     patience=5),
        ModelCheckpoint(
            auto_insert_metric_name=False,
            dirpath=path,
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
