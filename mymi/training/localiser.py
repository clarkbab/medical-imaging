import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchio.transforms import RandomAffine
from typing import Optional

from mymi import config
from mymi import dataset as ds
from mymi.loaders import Loader
from mymi.models.systems import Localiser

def train_localiser(
    dataset: str,
    num_gpu: int = 1,
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
    spacing = eval(set.params().spacing[0])
    train_loader = Loader.build(train_part, num_workers=8, spacing=spacing, transform=transform)
    val_loader = Loader.build(val_part, num_workers=8, shuffle=False)

    # Create checkpointing callback.
    path = os.path.join(config.directories.checkpoints, model_name, run_name)
    checkpoint = ModelCheckpoint(
        dirpath=path,
        every_n_epochs=1,
        monitor='val/loss')

    # Create logger.
    model = Localiser()
    logger = WandbLogger(
        project=model_name,
        log_model='all',
        name=run_name,
        save_dir=config.directories.wandb)
    logger.watch(model)

    # Perform training.
    trainer = Trainer(
        callbacks=[checkpoint],
        gpus=num_gpu,
        logger=logger,
        max_epochs=500,
        precision=16)
    trainer.fit(model, train_loader, val_loader)
