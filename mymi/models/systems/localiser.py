import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import SGD
from typing import List, Optional

from mymi.losses import DiceLoss
from mymi.metrics import batch_mean_dice, batch_mean_hausdorff_distance
from mymi.postprocessing import get_batch_largest_cc
from mymi import types

from ..networks import UNet

class Localiser(pl.LightningModule):
    def __init__(
        self,
        metrics: List[str] = [],
        spacing: Optional[types.ImageSpacing3D] = None):
        super().__init__()
        # Validate arguments.
        if 'hausdorff' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'Hausdorff' metric.")

        self._hausdorff_delay = 200
        self._loss = DiceLoss()
        self._metrics = metrics
        self._network = UNet()
        self._spacing = spacing

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = get_batch_largest_cc(pred)
        
        # Get bounding box.
        boxes = []
        for p in pred: 
            non_zero = np.argwhere(p != 0).astype(int)
            min = tuple(non_zero.min(axis=0))
            max = tuple(non_zero.max(axis=0))
            boxes.append((min, max))
        return boxes

    def training_step(self, batch, batch_idx):
        # Forward pass.
        x, labels = batch
        y = labels['Parotid_L']
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(np.bool)
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
        x = x.half()
        y = labels['Parotid_L']
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('val/dice', dice, on_epoch=True, sync_dist=True)

        if 'hausdorff' in self._metrics and batch_idx > self._hausdorff_delay:
            if y_hat.sum() > 0:
                hausdorff = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
                self.log('val/hausdorff', hausdorff, on_epoch=True, sync_dist=True)
