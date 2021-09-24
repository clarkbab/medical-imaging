import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch.optim import SGD
from typing import Dict, List, Optional

from mymi import config
from mymi.losses import DiceLoss
from mymi.metrics import batch_mean_dice, batch_mean_hausdorff_distance, batch_mean_surface_distance
from mymi.postprocessing import get_batch_largest_cc
from mymi import types

from ..networks import UNet

class Localiser(pl.LightningModule):
    def __init__(
        self,
        region: str,
        index_map: Optional[Dict[str, str]] = None,
        metrics: List[str] = [],
        spacing: Optional[types.ImageSpacing3D] = None):
        """
        args:
            index_map: maps from dataloader index to sample index. Used to identify metrics for 
                samples from validation batch.
            spacing: required to calculate the shape-based metrics.
        """
        super().__init__()
        # Validate arguments.
        if 'hausdorff' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'Hausdorff' metric.")

        self._hausdorff_delay = 50
        self._surface_delay = 50
        self._index_map = index_map
        self._loss = DiceLoss()
        self._log_args = {
            'on_epoch': True,
            'on_step': False,
        }
        self._metrics = metrics
        self._network = UNet()
        self._region = region
        self._spacing = spacing
        self.save_hyperparameters()

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str) -> pl.LightningModule:
        filename = f"{checkpoint}.ckpt"
        filepath = os.path.join(config.directories.checkpoints, model_name, run_name, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"Model '{model_name}' with run name '{run_name}' and checkpoint '{checkpoint}' not found.")
        return Localiser.load_from_checkpoint(filepath)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = pred.cpu().numpy().astype(np.bool)
        pred = get_batch_largest_cc(pred)

        return pred

    def training_step(self, batch, _):
        # Forward pass.
        x, labels = batch
        y = labels[self._region]
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(np.bool)
        self.log('train/loss', loss, **self._log_args)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('train/dice', dice, **self._log_args)

        if 'hausdorff' in self._metrics and self.global_step > self._hausdorff_delay:
            if y_hat.sum() > 0 and y.sum() > 0:
                hd, mean_hd = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
                self.log('train/hausdorff', hd, **self._log_args)
                self.log('train/average-hausdorff', mean_hd, **self._log_args)

        if 'surface' in self._metrics and self.global_step > self._surface_delay:
            if y_hat.sum() > 0 and y.sum() > 0:
                mean_sd, median_sd, std_sd, max_sd = batch_mean_surface_distance(y_hat, y, self._spacing)
                self.log('train/mean-surface', mean_sd, **self._log_args)
                self.log('train/median-surface', median_sd, **self._log_args)
                self.log('train/std-surface', std_sd, **self._log_args)
                self.log('train/max-surface', max_sd, **self._log_args)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        args:
            batch: the (input, label) pair of batched data.
            batch_idx: the batch index - increasing from 0 to len(loader). Note that this is only equivalent to the
                sample index because we use a batch size of 1 and don't shuffle validation data.
        """
        # Load sample description.
        sample_desc = self._index_map[batch_idx]

        # Forward pass.
        x, labels = batch
        y = labels[self._region]
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('val/loss', loss, **self._log_args, sync_dist=True)
        self.log(f"val/batch/loss/{sample_desc}", loss, on_epoch=False, on_step=True)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('val/dice', dice, **self._log_args, sync_dist=True)
            self.log(f"val/batch/dice/{sample_desc}", dice, on_epoch=False, on_step=True)

        if 'hausdorff' in self._metrics and self.global_step > self._hausdorff_delay:
            if y_hat.sum() > 0:
                hd, mean_hd = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
                self.log('val/hausdorff', hd, **self._log_args, sync_dist=True)
                self.log('val/average-hausdorff', mean_hd, **self._log_args, sync_dist=True)
                self.log(f"val/batch/hausdorff/{sample_desc}", hd, on_epoch=False, on_step=True)
                self.log(f"val/batch/average-hausdorff/{sample_desc}", mean_hd, on_epoch=False, on_step=True)

        if 'surface' in self._metrics and self.global_step > self._surface_delay:
            if y_hat.sum() > 0 and y.sum() > 0:
                mean_sd, median_sd, std_sd, max_sd = batch_mean_surface_distance(y_hat, y, self._spacing)
                self.log('val/mean-surface', mean_sd, **self._log_args, sync_dist=True)
                self.log('val/median-surface', median_sd, **self._log_args, sync_dist=True)
                self.log('val/std-surface', std_sd, **self._log_args, sync_dist=True)
                self.log('val/max-surface', max_sd, **self._log_args, sync_dist=True)
                self.log(f"val/batch/mean-surface/{sample_desc}", mean_sd, **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/median-surface/{sample_desc}", median_sd, **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/std-surface/{sample_desc}", std_sd, **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/max-surface/{sample_desc}", max_sd, **self._log_args, on_epoch=False, on_step=True)
