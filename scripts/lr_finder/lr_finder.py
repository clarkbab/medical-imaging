import os
import json
from pytorch_lightning import Trainer
from torchio.transforms import RandomAffine

from mymi import config
from mymi.loaders import Loader, MultiLoader
from mymi.losses import DiceWithFocalLoss
from mymi.models.systems import Segmenter, MultiSegmenter

datasets = 'PMCC-HN-REPLAN-LOC'
region = 'Brain'
test_fold = 0
halve_channels = True
n_gpus = 1
n_workers = 4

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

# Create data loaders.
train_loader, val_loader, _ = MultiLoader.build_loaders(datasets, n_workers=n_workers, region=region, test_fold=test_fold, transform=transform)

# Create model.
model = MultiSegmenter(
    region,
    loss_fn,
    halve_channels=halve_channels,
    metrics=['dice'],
    n_gpus=n_gpus)

trainer = Trainer(
    accelerator='gpu',
    devices=1,
    precision=16)

lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, min_lr=1e-8, max_lr=1e3, num_training=1000, early_stop_threshold=None)
print(lr_finder.suggestion(skip_begin=20))      # For some reason, LR decreases sharply for very low learning rate (~1e-8).

filepath = os.path.join(config.directories.files, 'results.json')
with open(filepath, 'w') as f:
    f.write(json.dumps(lr_finder.results))
