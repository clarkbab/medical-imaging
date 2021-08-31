import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from torchio.transforms import RandomAffine
from typing import Optional

from mymi import config
from mymi import dataset as ds
from mymi.loaders import Loader
from mymi.models.systems import Localiser
from mymi import types

def train_localiser(
    model_name: str,
    run_name: str,
    dataset: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    num_workers: int = 1,
    regions: types.PatientRegions = 'all',
    use_logger: bool = False) -> None:

    # Load partitions.
    set = ds.get(dataset, 'processed')
    train_part = set.partition('train')
    val_part = set.partition('validation')

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
    spacing = eval(set.params().spacing[0])
    train_loader = Loader.build(train_part, num_workers=num_workers, regions=regions, spacing=spacing, transform=transform)
    val_loader = Loader.build(val_part, num_workers=num_workers, regions=regions, shuffle=False)

    # Create model.
    metrics = ['dice', 'hausdorff']
    model = Localiser(
        region=regions,
        metrics=metrics,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            project=model_name,
            name=run_name,
            save_dir=config.directories.wandb)
        logger.watch(model)
    else:
        logger = None

    # Create callbacks.
    path = os.path.join(config.directories.checkpoints, model_name, run_name)
    callbacks = [
        # EarlyStopping(
        #     monitor='val/loss',
        #     patience=5),
        ModelCheckpoint(
            dirpath=path,
            every_n_epochs=1,
            monitor='val/loss',
            save_top_k=3)
    ]

    # Perform training.
    trainer = Trainer(
        accelerator='ddp',
        callbacks=callbacks,
        gpus=list(range(num_gpus)),
        logger=logger,
        max_epochs=200,
        num_nodes=num_nodes,
        num_sanity_val_steps=0,
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=16)
    trainer.fit(model, train_loader, val_loader)
