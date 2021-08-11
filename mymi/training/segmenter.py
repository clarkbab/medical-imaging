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
from mymi.loaders import PatchLoader
from mymi.models.systems import Segmenter

def train_segmenter(
    dataset: str,
    num_gpus: int = 1,
    num_nodes: int = 1,
    num_workers: int = 1,
    run_name: Optional[str] = None,
    use_logger: bool = False) -> None:
    model_name = 'segmenter-pl'

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
    patch_size = (128, 128, 96)
    region = 'Parotid_L'
    spacing = eval(set.params().spacing[0])
    train_loader = PatchLoader.build(train_part, patch_size, region, num_workers=num_workers, spacing=spacing, transform=transform)
    val_loader = PatchLoader.build(val_part, patch_size, region, num_workers=num_workers, shuffle=False)

    # Create model.
    metrics = ['dice', 'hausdorff']
    model = Segmenter(
        metrics=metrics,
        spacing=spacing)

    # Create logger.
    if use_logger:
        logger = WandbLogger(
            project=model_name,
            log_model='all',
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
        gpus=num_gpus,
        logger=logger,
        max_epochs=200,
        num_nodes=num_nodes,
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=16)
    trainer.fit(model, train_loader, val_loader)
