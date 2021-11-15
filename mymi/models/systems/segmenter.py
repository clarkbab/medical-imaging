import numpy as np
import os
import pytorch_lightning as pl
from scipy.ndimage import center_of_mass
import torch
from torch import nn
from torch.optim import SGD
from typing import Dict, List, Optional
import wandb

from mymi import config
from mymi.losses import DiceLoss
from mymi.metrics import batch_mean_dice, batch_mean_distances
from mymi.postprocessing import get_batch_largest_cc
from mymi import types

from ..networks import UNet

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module = DiceLoss(),
        metrics: List[str] = [],
        predict_logits: bool = False,
        spacing: Optional[types.ImageSpacing3D] = None):
        super().__init__()
        if 'distances' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'distances' metric.")
        self._distances_delay = 50
        self._distances_interval = 20
        self._surface_delay = 50
        self._surface_interval = 20
        self._loss = loss
        self._max_images = 50
        self._metrics = metrics
        self._network = UNet()
        self._predict_logits = predict_logits
        self._spacing = spacing

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str,
        **kwargs: Dict) -> pl.LightningModule:
        filename = f"{checkpoint}.ckpt"
        filepath = os.path.join(config.directories.checkpoints, model_name, run_name, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"Model '{model_name}' with run name '{run_name}' and checkpoint '{checkpoint}' not found.")
        return Segmenter.load_from_checkpoint(filepath, **kwargs)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        if self._predict_logits:
            return pred

        # Apply thresholding.
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = pred.cpu().numpy().astype(np.bool)
        pred = get_batch_largest_cc(pred)

        return pred

    def training_step(self, batch, _):
        # Forward pass.
        _, x, y = batch
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('train/loss', loss, on_epoch=True)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('train/dice', dice, on_epoch=True)

        # if 'hausdorff' in self._metrics and self.global_step > self._hausdorff_delay:
        #     if y_hat.sum() > 0:
        #         hausdorff = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
        #         self.log('train/hausdorff', hausdorff, on_epoch=True)

        # if 'surface' in self._metrics and self.global_step > self._surface_delay:
        #     if y_hat.sum() > 0 and y.sum() > 0:
        #         mean_sd, median_sd, std_sd, max_sd = batch_mean_symmetric_surface_distance(y_hat, y, self._spacing)
        #         self.log('train/mean-surface', mean_sd, **self._log_args)
        #         self.log('train/median-surface', median_sd, **self._log_args)
        #         self.log('train/std-surface', std_sd, **self._log_args)
        #         self.log('train/max-surface', max_sd, **self._log_args)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        if not batch:
            raise ValueError(f"Batch is none")
        descs, x, y = batch
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('val/loss', loss, on_epoch=True, sync_dist=True)
        self.log(f"val/batch/loss/{sample_desc}", loss, on_epoch=False, on_step=True)

        if 'dice' in self._metrics:
            dice = batch_mean_dice(y_hat, y)
            self.log('val/dice', dice, on_epoch=True, sync_dist=True)
            self.log(f"val/batch/dice/{sample_desc}", dice, on_epoch=False, on_step=True)

        if 'distances' in self._metrics and self.global_step > self._distances_delay and self.current_epoch % self._distances_interval == 0:
            if y_hat.sum() > 0 and y.sum() > 0:
                dists = batch_mean_distances(y_hat, y, self._spacing)
                self.log('val/assd', dists['assd'], **self._log_args, sync_dist=True)
                self.log('val/surface-hd', dists['surface-hd'], **self._log_args, sync_dist=True)
                # self.log('val/surface-ahd', dists['surface-ahd'], **self._log_args, sync_dist=True)
                self.log('val/surface-95hd', dists['surface-95hd'], **self._log_args, sync_dist=True)
                self.log('val/voxel-hd', dists['voxel-hd'], **self._log_args, sync_dist=True)
                # self.log('val/voxel-ahd', dists['voxel-ahd'], **self._log_args, sync_dist=True)
                self.log('val/voxel-95hd', dists['voxel-95hd'], **self._log_args, sync_dist=True)
                self.log(f"val/batch/assd/{descs[0]}", dists['assd'], on_epoch=False, on_step=True)
                self.log(f"val/batch/surface-hd/{descs[0]}", dists['surface-hd'], on_epoch=False, on_step=True)
                # self.log(f"val/batch/surface-ahd/{descs[0]}", dists['surface-ahd'], on_epoch=False, on_step=True)
                self.log(f"val/batch/surface-95hd/{descs[0]}", dists['surface-95hd'], **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/voxel-hd/{descs[0]}", dists['voxel-hd'], **self._log_args, on_epoch=False, on_step=True)
                # self.log(f"val/batch/voxel-ahd/{descs[0]}", dists['voxel-ahd'], **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/voxel-95hd/{descs[0]}", dists['voxel-95hd'], **self._log_args, on_epoch=False, on_step=True)

        # Log predictions.
        if self.logger:
            class_labels = {
                1: 'foreground'
            }
            for i, desc in enumerate(descs):
                if batch_idx < self._max_images:
                    # Get images.
                    x_vol, y_vol, y_hat_vol = x[i, 0].cpu().numpy(), y[i], y_hat[i]

                    # Get centre of extent of ground truth.
                    centre = get_extent_centre(y_vol)

                    for axis, centre_ax in enumerate(centre):
                        # Get slices.
                        slices = tuple([centre_ax if i == axis else slice(0, x_vol.shape[i]) for i in range(0, len(x_vol.shape))])
                        x_img, y_img, y_hat_img = x_vol[slices], y_vol[slices], y_hat_vol[slices]

                        # Fix orientation.
                        if axis == 0 or axis == 1:
                            x_img = np.rot90(x_img)
                            y_img = np.rot90(y_img)
                            y_hat_img = np.rot90(y_hat_img)
                        elif axis == 2:
                            x_img = np.transpose(x_img)
                            y_img = np.transpose(y_img) 
                            y_hat_img = np.transpose(y_hat_img)

                        # Send image.
                        image = wandb.Image(
                            x_img,
                            caption=desc,
                            masks={
                                'ground_truth': {
                                    'mask_data': y_img,
                                    'class_labels': class_labels
                                },
                                'predictions': {
                                    'mask_data': y_hat_img,
                                    'class_labels': class_labels
                                }
                            }
                        )
                        title = f'{desc}:axis:{axis}'
                        self.logger.experiment.log({ title: image })
