import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch
from IPython.display import display
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import wandb

from mymi import config
from mymi.loaders import MultiLoader
from mymi.loaders.augmentation import get_transforms
from mymi.loaders.hooks import naive_crop
from mymi.models.systems import MultiSegmenter

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

# Create data loaders.
transform_train, transform_val = get_transforms(thresh_high=None, thresh_low=None, use_stand=False, use_thresh=False)
resolution = '444'
dataset = f'MICCAI-2015-{resolution}'
regions = ['Bone_Mandible']
train_loader, val_loader, _ = MultiLoader.build_loaders(dataset, data_hook=naive_crop, region=regions, transform_train=transform_train, transform_val=transform_val, use_split_file=True)

# Create model.
model = MultiSegmenter(
    lr_init=1e-4,
    metrics=['dice'],
    n_gpus=1,
    n_split_channels=2,
    region=regions)

# Create logger.
offline = True
logger = WandbLogger(project='Test', name='test', save_dir=config.directories.reports, offline=offline)
logger.watch(model, log='all')

# Initialise trainer.
trainer = Trainer(
    accelerator="auto",
    devices=1,
    logger=logger,
    max_epochs=100,
)

# Train model.
trainer.fit(model, train_loader, val_loader)
