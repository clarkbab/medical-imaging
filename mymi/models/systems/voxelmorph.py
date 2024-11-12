import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple
import wandb

from mymi import config
from mymi.losses.voxelmorph import Grad, NCC
from mymi.models import replace_ckpt_alias
from mymi import types

from ..networks import VxmDense

class Voxelmorph(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.__loss = lambda y_true, y_pred: NCC().loss(y_true, y_pred) + 0.01 * Grad('l2', loss_mult=2).loss(y_true, y_pred)
        self._log_args = {
            'on_epoch': True,
            'on_step': False,
        }
        self.__max_image_batches = 2
        self.__network = VxmDense()
        self.save_hyperparameters(ignore=['loss'])

    @property
    def network(self) -> nn.Module:
        return self._network

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
                raise ValueError(f"Can't load voxelmorph ('{model_name}', '{run_name}'), has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model_name, run_name, checkpoint = replace_ckpt_alias((model_name, run_name, checkpoint))
        filepath = os.path.join(config.directories.models, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Checkpoint '{checkpoint}' not found for voxelmorph run '{model_name}:{run_name}'.")
        localiser = Localiser.load_from_checkpoint(filepath, **kwargs)
        localiser._name = (model_name, run_name, checkpoint)
        return localiser

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def forward(self, x_fixed, x_moving):
        return self._network(x_fixed, x_moving)

    def training_step(self, batch, _):
        # Forward pass.
        _, x_fixed, x_moving = batch
        y_moved, y_warp = self._network(x_fixed, x_moving)
        loss = self.__loss(y_moved, y_warp)

        # Log metrics.
        self.log('train/loss', loss, **self._log_args)

        return loss

    def validation_step(self, batch, batch_idx):
        descs, x_fixed, x_moving = batch
        y_moved, y_warp = self._network(x_fixed, x_moving)
        loss = self.__loss(y_moved, y_warp)

        # Log metrics.
        self.log('val/loss', loss, **self._log_args, sync_dist=True)

        # Log predictions.
        if self.logger:
            for i, desc, x_f, x_m, y_m, y_w in zip(descs, x_fixed, x_moving, y_moved, y_warp):
                if batch_idx < self.__max_image_batches:
                    # Show axial slices.
                    idx = x_f.shape[2] // 2
                    x_f, x_m, y_m, y_w = x_f[:, :, idx], x_m[:, :, idx], y_m[:, :, idx], y_w[:, :, idx]

                    # Fix orientation.
                    x_f, x_m, y_m, y_w = np.transpose(x_f), np.transpose(x_m), np.transpose(y_m), np.transpose(y_w)

                    # Add images into a single image.
                    img = np.concatenate([x_f, x_m, y_m], axis=1)

                    # Send image.
                    image = wandb.Image(
                        img,
                        caption=desc,
                    )
                    title = f'{desc}:axial'
                    self.logger.experiment.log({ title: image })
