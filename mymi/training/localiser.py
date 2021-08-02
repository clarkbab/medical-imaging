import pytorch_lightning as pl
from pytorch_lightning import Trainer
from typing import Optional

from mymi import dataset as ds
from mymi.loaders import Loader
from mymi.models.systems import Localiser

def train_localiser(
    dataset: str,
    gpus: int = 1):

    # Load partitions.
    set = ds.get(dataset, 'processed')
    train_part = set.partition('train')
    val_part = set.partition('validation')

    # Create data loaders.
    train_loader = Loader.build(train_part)
    val_loader = Loader.build(val_part, num_workers=8, shuffle=False)

    # Perform training.
    model = Localiser()
    trainer = Trainer(gpus=gpus, precision=16)
    trainer.fit(model, train_loader, val_loader)
