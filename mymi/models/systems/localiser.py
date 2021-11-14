import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD
from typing import Dict, List, Optional, Tuple
import wandb

from mymi import config
from mymi.losses import DiceLoss
from mymi.metrics import batch_mean_dice, batch_mean_distances
from mymi.postprocessing import get_batch_largest_cc, get_extent_centre
from mymi import types

from ..networks import UNet

class Localiser(pl.LightningModule):
    def __init__(
        self,
        region: str,
        index_map: Optional[Dict[str, str]] = None,
        loss: nn.Module = DiceLoss(),
        metrics: List[str] = [],
        predict_logits: bool = False,
        pretrained_model: Optional[pl.LightningModule] = None,
        spacing: Optional[types.ImageSpacing3D] = None):
        """
        args:
            index_map: maps from dataloader index to sample index. Used to identify metrics for 
                samples from validation batch.
            spacing: required to calculate the shape-based metrics.
        """
        super().__init__()
        if 'distances' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'distances' metric.")
        self._distances_delay = 50
        self._distances_interval = 20
        self._index_map = index_map
        self._loss = loss
        self._log_args = {
            'on_epoch': True,
            'on_step': False,
        }
        self._max_images = 50
        self._metrics = metrics
        self._network = UNet()
        self._region = region
        self._predict_logits = predict_logits
        self._spacing = spacing
        self.save_hyperparameters()

        # Replace pretrained layers and freeze.
        if pretrained_model is not None:
            self._network.transfer_encoder(pretrained_model)

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str,
        **kwargs: Dict) -> pl.LightningModule:
        # Load model.
        model_name, run_name, checkpoint = Localiser.replace_best(model_name, run_name, checkpoint)
        filepath = os.path.join(config.directories.checkpoints, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Model '{model_name}' with run name '{run_name}' and checkpoint '{checkpoint}' not found.")
        return Localiser.load_from_checkpoint(filepath, **kwargs)

    @staticmethod
    def replace_best(
        model_name: str,
        run_name: str,
        checkpoint: str) -> Tuple[str, str, str]:
        # Find best checkpoint.
        if checkpoint == 'BEST': 
            checkpath = os.path.join(config.directories.checkpoints, model_name, run_name)
            checkpoint = list(sorted(os.listdir(checkpath)))[-1].replace('.ckpt', '')

        return (model_name, run_name, checkpoint)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        if self._predict_logits:
            pred = pred.cpu().numpy()
            return pred

        # Apply thresholding.
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

        # if 'hausdorff' in self._metrics and self.global_step > self._hausdorff_delay:
        #     if y_hat.sum() > 0 and y.sum() > 0:
        #         hd, mean_hd = batch_mean_hausdorff_distance(y_hat, y, self._spacing)
        #         self.log('train/hausdorff', hd, **self._log_args)
        #         self.log('train/average-hausdorff', mean_hd, **self._log_args)

        # if 'surface' in self._metrics and self.global_step > self._surface_delay:
        #     if y_hat.sum() > 0 and y.sum() > 0:
        #         mean_sd, median_sd, std_sd, max_sd = batch_mean_symmetric_surface_distance(y_hat, y, self._spacing)
        #         self.log('train/mean-surface', mean_sd, **self._log_args)
        #         self.log('train/median-surface', median_sd, **self._log_args)
        #         self.log('train/std-surface', std_sd, **self._log_args)
        #         self.log('train/max-surface', max_sd, **self._log_args)

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
                self.log(f"val/batch/assd/{sample_desc}", dists['assd'], on_epoch=False, on_step=True)
                self.log(f"val/batch/surface-hd/{sample_desc}", dists['surface-hd'], on_epoch=False, on_step=True)
                # self.log(f"val/batch/surface-ahd/{sample_desc}", dists['surface-ahd'], on_epoch=False, on_step=True)
                self.log(f"val/batch/surface-95hd/{sample_desc}", dists['surface-95hd'], **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/voxel-hd/{sample_desc}", dists['voxel-hd'], **self._log_args, on_epoch=False, on_step=True)
                # self.log(f"val/batch/voxel-ahd/{sample_desc}", dists['voxel-ahd'], **self._log_args, on_epoch=False, on_step=True)
                self.log(f"val/batch/voxel-95hd/{sample_desc}", dists['voxel-95hd'], **self._log_args, on_epoch=False, on_step=True)

        # Log prediction.
        if batch_idx < self._max_images:
            class_labels = {
                0: 'background',
                1: 'foreground'
            }
            x_vol, y_vol, y_hat_vol = x[0, 0].cpu().numpy(), y[0], y_hat[0]
            coe = list(np.round(get_extent_centre(y_vol)).astype(int))
            for axis, coe_ax in enumerate(coe):
                slices = tuple([coe_ax if i == axis else slice(0, x_vol.shape[i]) for i in range(0, len(x_vol.shape))])
                x_img, y_img, y_hat_img = x_vol[slices], y_vol[slices], y_hat_vol[slices]

                # Fix orientation.
                if axis == 0 or axis == 1:
                    x_img = np.rot90(y_img, k=-1)
                    y_img = np.rot90(y_hat_img, k=-1)
                    y_hat_img = np.rot90(x_img, k=-1)
                elif axis == 2:
                    x_img = np.transpose(x_img)
                    y_img = np.transpose(y_img) 
                    y_hat_img = np.transpose(y_hat_img)

                # Send image.
                image = wandb.Image(
                    x_img,
                    caption=sample_desc,
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
                title = f'{sample_desc}:axis:{axis}'
                self.logger.experiment.log({ title: image })
