import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchio.transforms import RandomAffine
from typing import Optional

from mymi import config
from mymi import dataset as ds
from mymi.loaders import Loader
from mymi.models.systems import Localiser

def train_localiser(
    dataset: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    num_workers: int = 1,
    run_name: Optional[str] = None) -> None:
    model_name = 'localiser-pl'

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
    regions = 'Parotid_L'
    spacing = eval(set.params().spacing[0])
    train_loader = Loader.build(train_part, num_workers=num_workers, regions=regions, spacing=spacing, transform=transform)
    val_loader = Loader.build(val_part, num_workers=num_workers, regions=regions, shuffle=False)

    # Create checkpointing callback.
    path = os.path.join(config.directories.checkpoints, model_name, run_name)
    checkpoint = ModelCheckpoint(
        dirpath=path,
        every_n_epochs=1,
        monitor='val/loss')

    # Create model.
    metrics = ['dice', 'hausdorff']
    model = Localiser(metrics=metrics)

    # Create logger.
    logger = WandbLogger(
        project=model_name,
        log_model='all',
        name=run_name,
        save_dir=config.directories.wandb)
    logger.watch(model)

    # Create callbacks.
    callbacks = [
        EarlyStopping('val/loss'),
        ModelCheckpoint(
            dirpath=path,
            every_n_epochs=1,
            monitor='val/loss')
    ]

    # Perform training.
    trainer = Trainer(
        callbacks=callbacks,
        gpus=num_gpus,
        logger=logger,
        max_epochs=500,
        num_nodes=num_nodes,
        precision=16)
    trainer.fit(model, train_loader, val_loader)
