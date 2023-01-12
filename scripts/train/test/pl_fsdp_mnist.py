import os

import pandas as pd
import seaborn as sn
import torch
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from functools import partial

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 4096

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            torch.nn.Linear(28 * 28, 10),
            torch.nn.ReLU()
        )

    def configure_sharded_model(self):
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e4)
        # 'auto_wrap_policy' causes recursive layer wrapping using custom policy, 'device_id' ensures sharding happens
        # on GPU.
        self.model = FSDP(self.model, auto_wrap_policy=auto_wrap_policy, device_id=torch.distributed.get_rank())

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        print(f'[{torch.distributed.get_rank()} - {torch.cuda.current_device()}] Memory allocated: {torch.cuda.memory_allocated()}')
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.trainer.model.parameters(), lr=0.02)

# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# Initialize a trainer
trainer = Trainer(
    accelerator="gpu",
    devices=2,
    max_epochs=5,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    strategy='fsdp_native'
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)
print('done!')
