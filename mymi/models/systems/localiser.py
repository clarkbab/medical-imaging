import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import SGD
from typing import List, Optional

from mymi.metrics import batch_mean_dice, batch_mean_hausdorff_distance
from mymi.postprocessing import get_largest_cc
from mymi import types

from ..networks import UNet

class Localiser(pl.LightningModule):
    def __init__(
        self,
        metrics: List[str] = [],
        spacing: Optional[types.ImageSpacing3D] = None):
        super().__init__()
        self._hausdorff_delay = 200
        self._metrics = metrics
        self._network = UNet()
        self._spacing = spacing

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        
        # Apply postprocessing.
        pred = get_largest_cc(pred)
        
        # Get bounding box.
        non_zero = np.argwhere(pred != 0).astype(int)
        min = tuple(non_zero.min(axis=0))
        max = tuple(non_zero.max(axis=0))
        bounding_box = (min, max)
        return bounding_box

    def training_step(self, batch, batch_idx):
        # Forward pass.
        x, labels = batch
        y = labels['Parotid_L']
        y_hat = self._network(x)
        loss = self._loss_fn(y_hat, y)

        # Log metrics.
        self.log('train/loss', loss, on_epoch=True)
        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('train/dice', dice, on_epoch=True)

        if 'hausdorff' in self._metrics and batch_idx > self._hausdorff_delay:
            if y_hat.sum() > 0:
                hausdorff = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
                self.log('train/hausdorff', hausdorff, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        x, labels = batch
        y = labels['Parotid_L']
        y_hat = self._network(x)
        loss = self._loss_fn(y_hat, y)

        # Log metrics.
        self.log('validation/loss', loss, on_epoch=True)
        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('validation/dice', dice, on_epoch=True)

        if 'hausdorff' in self._metrics and batch_idx > self._hausdorff_delay:
            if y_hat.sum() > 0:
                hausdorff = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
                self.log('validation/hausdorff', hausdorff, on_epoch=True)
