import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import SGD
from typing import Dict, List, Optional, Tuple
import wandb

from mymi import config
from mymi.geometry import get_extent_centre
from mymi.losses import DiceLoss
from mymi.metrics import batch_mean_dice, batch_mean_all_distances
from mymi.models import replace_ckpt_alias
from mymi.postprocessing import largest_cc_4D
from mymi import types

from ..networks import MultiUNet3D, UNet3D

MODE = 0

class Localiser(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module = DiceLoss(),
        metrics: List[str] = [],
        predict_logits: bool = False,
        pretrained: Optional[pl.LightningModule] = None,
        spacing: Optional[types.Spacing3D] = None):
        super().__init__()
        if 'distances' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'distances' metric.")
        self._distances_delay = 50
        self._distances_interval = 20
        self._loss = loss
        self._log_args = {
            'on_epoch': True,
            'on_step': False,
        }
        self._max_image_batches = 30
        self._metrics = metrics
        self._name = None
        pretrained_model = pretrained.network if pretrained else None
        self._network = UNet3D(pretrained_model=pretrained_model)
        # self._network = MultiUNet3D(2, n_ckpts=22, halve_channels=False)
        self._predict_logits = predict_logits
        self._spacing = spacing
        self.save_hyperparameters(ignore=['loss'])

    @property
    def network(self) -> nn.Module:
        return self._network

    @property
    def name(self) -> Optional[Tuple[str, str, str]]:
        return self._name

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str,
        check_epochs: bool = True,
        n_epochs: Optional[int] = np.inf,
        **kwargs: Dict) -> pl.LightningModule:
        # Check that model training has finished.
        if check_epochs:
            last_model = replace_ckpt_alias((model_name, run_name, 'last'))
            filepath = os.path.join(config.directories.models, last_model[0], last_model[1], f'{last_model[2]}.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            n_epochs_complete = state['epoch'] + 1
            if n_epochs_complete < n_epochs:
                raise ValueError(f"Can't load localiser ('{model_name}', '{run_name}'), has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model_name, run_name, checkpoint = replace_ckpt_alias((model_name, run_name, checkpoint))
        filepath = os.path.join(config.directories.models, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Checkpoint '{checkpoint}' not found for localiser run '{model_name}:{run_name}'.")
        localiser = Localiser.load_from_checkpoint(filepath, **kwargs)
        localiser._name = (model_name, run_name, checkpoint)
        return localiser

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def print_batch_norm_layers(self):
        self._network.print_batch_norm_layers()

    def forward(self, x):
        # Get prediction.
        pred = self._network(x)
        if self._predict_logits:
            pred = pred.cpu().numpy()
            return pred

        # Apply thresholding.
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = pred.cpu().numpy().astype(np.bool_)
        pred = largest_cc_4D(pred)

        return pred

    def training_step(self, batch, _):
        # Forward pass.
        if MODE == 0:
            _, x, y = batch
        elif MODE == 1:
            _, x, y, _, _ = batch
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(np.bool_)
        self.log('train/loss', loss, **self._log_args)

        if 'dice' in self._metrics:
            if MODE == 1:
                y = y.argmax(axis=1).astype(np.bool_)
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
            batch: the (desc, input, label) pair of batched data.
        """
        # Forward pass.
        if MODE == 0:
            descs, x, y = batch
        elif MODE == 1:
            descs, x, y, _, _ = batch
        y_hat = self._network(x)
        loss = self._loss(y_hat, y)

        # Log metrics.
        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy().astype(bool)
        self.log('val/loss', loss, **self._log_args, sync_dist=True)
        self.log(f"val/batch/loss/{descs[0]}", loss, on_epoch=False, on_step=True)

        if 'dice' in self._metrics:
            if MODE == 1:
                y = y.argmax(axis=1).astype(np.bool_)
            dice = batch_mean_dice(y_hat, y)
            self.log('val/dice', dice, **self._log_args, sync_dist=True)
            self.log(f"val/batch/dice/{descs[0]}", dice, on_epoch=False, on_step=True)

        # Log predictions.
        if self.logger:
            class_labels = {
                1: 'foreground'
            }
            for i, desc in enumerate(descs):
                if batch_idx < self._max_image_batches:
                    # Get images.
                    x_vol, y_vol, y_hat_vol = x[i, 0].cpu().numpy(), y[i], y_hat[i]

                    # Get centre of extent of ground truth.
                    centre = get_extent_centre(y_vol)
                    if centre is None:
                        # Empty ground truth.
                        continue

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
