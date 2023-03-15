import json
import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from mymi import config

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

model = MNISTModel()
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = Trainer(
    accelerator='gpu',
    callbacks=[lr_monitor],
    devices=1,
    precision=16)

lr_finder = trainer.tuner.lr_find(model, train_dataloaders=train_loader, min_lr=1e-8, max_lr=1e2, num_training=1000)
print(lr_finder.suggestion())

filepath = os.path.join(config.directories.files, 'results-mnist.json')
with open(filepath, 'w') as f:
    f.write(json.dumps(lr_finder.results))

