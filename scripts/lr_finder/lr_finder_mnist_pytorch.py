import json
import os
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch_lr_finder import LRFinder

from mymi import config

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

model = MNISTModel()
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-8)
lr_finder = LRFinder(model, optimiser, loss_fn, device='cuda')
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

lr_finder.range_test(train_loader, end_lr=1e2, num_iter=1000)

filepath = os.path.join(config.directories.files, 'results-mnist-pytorch.json')
with open(filepath, 'w') as f:
    f.write(json.dumps(lr_finder.results))

