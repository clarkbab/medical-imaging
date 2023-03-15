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
from mymi.loaders import MultiLoader
from mymi.losses import DiceLoss
from mymi.models.networks import MultiUNet3D

dataset = 'PMCC-HN-REPLAN-LOC'
region = 'Brain'
test_fold = 0

model = MultiUNet3D(n_output_channels=2)
loss_fn = DiceLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-8)
lr_finder = LRFinder(model, optimiser, loss_fn, device='cuda')
train_loader, val_loader, _ = MultiLoader.build_loaders(dataset, region=region, test_fold=test_fold)

class TrainIter

If your DataLoader returns e.g. dict, or other non standard output, intehit from TrainDataLoaderIter,
        redefine method `inputs_labels_from_batch` so that it outputs (inputs, lables) data:
            >>> import torch_lr_finder
            >>> class TrainIter(torch_lr_finder.TrainDataLoaderIter):
            >>>     def inputs_labels_from_batch(self, batch_data):
            >>>         return (batch_data['user_features'], batch_data['user_history']), batch_data['y_labels']
            >>> train_data_iter = TrainIter(train_dl)
            >>> finder = torch_lr_finder.LRFinder(model, optimizer, partial(model._train_loss, need_one_hot=False))
            >>> finder.range_test(train_data_iter, end_lr=10, num_iter=300, diverge_th=10)

lr_finder.range_test(train_loader, end_lr=1e2, num_iter=100)

filepath = os.path.join(config.directories.files, 'results-pytorch.json')
with open(filepath, 'w') as f:
    f.write(json.dumps(lr_finder.results))

