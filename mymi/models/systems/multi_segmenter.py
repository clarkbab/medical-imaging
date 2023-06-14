import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.nn.functional as F
from typing import Dict, List, Optional, OrderedDict, Tuple
from wandb import Image

from mymi import config
from mymi.geometry import get_extent_centre
from mymi import logging
from mymi.losses import DiceWithFocalLoss
from mymi.metrics import dice
from mymi.models import replace_ckpt_alias
from mymi.models.networks import MultiUNet3D
from mymi.regions import region_to_list
from mymi import types

LOG_ON_EPOCH = True
LOG_ON_STEP = False

class MultiSegmenter(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module = DiceWithFocalLoss(),
        lr_init: float = 1e-3,
        max_image_batches: int = 2,
        metrics: List[str] = [],
        region: types.PatientRegions = 'all',
        use_lr_scheduler: bool = False,
        val_image_interal: int = 10,
        weight_decay: float = 0,
        **kwargs):
        super().__init__()
        self.__loss = loss
        self.lr = lr_init        # 'self.lr' is default key that LR finder looks for on module.
        self.__name = None
        self.__max_image_batches = max_image_batches
        self.__metrics = metrics
        self.__regions = region_to_list(region)
        self.__n_output_channels = len(self.__regions) + 1
        self.__network = MultiUNet3D(self.__n_output_channels, **kwargs)
        self.__use_lr_scheduler = use_lr_scheduler
        self.__val_image_interval = val_image_interal
        self.__weight_decay = weight_decay

        # Create channel -> region map.
        self.__channel_region_map = { 0: 'background' }
        for i, region in enumerate(self.__regions):
            self.__channel_region_map[i + 1] = region 

    @property
    def name(self) -> Optional[Tuple[str, str, str]]:
        return self.__name

    @property
    def network(self) -> nn.Module:
        return self.__network

    @staticmethod
    def load(
        model_name: str,
        run_name: str,
        checkpoint: str,
        check_epochs: bool = True,
        **kwargs: Dict) -> pl.LightningModule:
        # Check that model training has finished.
        if check_epochs:
            filepath = os.path.join(config.directories.models, model_name, run_name, 'last.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            n_epochs = 3000
            if state['epoch'] < n_epochs - 1:
                raise ValueError(f"Can't load multi-segmenter ('{model_name}','{run_name}','{checkpoint}') - hasn't completed {n_epochs} epochs training.")

        # Load model.
        model_name, run_name, checkpoint = replace_ckpt_alias((model_name, run_name, checkpoint))
        filepath = os.path.join(config.directories.models, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Segmenter '{model_name}' with run name '{run_name}' and checkpoint '{checkpoint}' not found.")

        # Update keys by adding '_Segmenter_' prefix if required.
        checkpoint_data = torch.load(filepath, map_location=torch.device('cpu'))
        pairs = []
        update = False
        for k, v in checkpoint_data['state_dict'].items():
            # Get new key.
            if k.startswith('_Segmenter__MultiSegmenter__network'):
                update = True
                new_k = k.replace('_Segmenter__MultiSegmenter__network', '_MultiSegmenter__network')
            else:
                new_k = k
            pairs.append((new_k, v))
        checkpoint_data['state_dict'] = OrderedDict(pairs)
        if update:
            logging.info(f"Updating checkpoint keys for model '{(model_name, run_name, checkpoint)}'.")
            torch.save(checkpoint_data, filepath)

        # Load checkpoint.
        segmenter = MultiSegmenter.load_from_checkpoint(filepath, **kwargs)
        segmenter.__name = (model_name, run_name, checkpoint)
        return segmenter

    def configure_optimizers(self):
        self.__optimiser = Adam(self.parameters(), lr=self.lr, weight_decay=self.__weight_decay) 
        opt = {
            'optimizer': self.__optimiser,
            'monitor': 'val/loss'
        }
        if self.__use_lr_scheduler:
            # opt['lr_scheduler'] = MultiStepLR(self.__optimiser, [120])
            opt['lr_scheduler'] = ReduceLROnPlateau(self.__optimiser, factor=0.5, patience=200, verbose=True)

        return opt

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        return self.__network(x)

    def training_step(self, batch, _):
        # Forward pass.
        desc, x, y, mask, weights = batch
        y_hat = self.forward(x)
        loss = self.__loss(y_hat, y, mask, weights)
        if np.isnan(loss.item()):
            print(desc)
            # names = ['x', 'y', 'mask', 'weights', 'y_hat']
            # arrays = [x, y, mask, weights, y_hat]
            # for name, arr in zip(names, arrays):
            #     filepath = os.path.join(config.directories.temp, f'{name}.npy')
            #     np.save(filepath, arr.detach().cpu().numpy())
            # filepath = os.path.join(config.directories.temp, 'model.ckpt')
            # torch.save(self.__network.state_dict(), filepath)
        else:
            self.log('train/loss', loss, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

        # Convert pred to binary mask.
        y_hat = F.one_hot(y_hat.argmax(axis=1), num_classes=y.shape[1]).moveaxis(-1, 1)  # 'F.one_hot' adds new axis last, move to second place.
        y_hat = y_hat.cpu().numpy().astype(bool)

        # Report metrics.
        y = y.cpu().numpy()
        if 'dice' in self.__metrics:
            # Get mean dice score per-channel.
            for i in range(self.__n_output_channels):
                region = self.__channel_region_map[i]
                dice_scores = []
                for b in range(y.shape[0]):     # Batch items.
                    if mask[b, i]:
                        y_i = y[b, i]   
                        y_hat_i = y_hat[b, i]
                        dice_score = dice(y_hat_i, y_i)
                        dice_scores.append(dice_score)

                if len(dice_scores) > 0:
                    mean_dice = np.mean(dice_scores)
                    self.log(f'train/dice/{region}', mean_dice, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        descs, x, y, mask, weights = batch
        y_hat = self.forward(x)
        loss = self.__loss(y_hat, y, mask, weights)
        self.log('val/loss', loss, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

        # Convert pred to binary mask.
        y = y.cpu().numpy()
        y_hat = F.one_hot(y_hat.argmax(axis=1), num_classes=y.shape[1]).moveaxis(-1, 1)  # 'F.one_hot' adds new axis last, move to second place.
        y_hat = y_hat.cpu().numpy().astype(bool)

        if 'dice' in self.__metrics:
            # Get mean dice score per-channel.
            for i in range(self.__n_output_channels):
                region = self.__channel_region_map[i]
                dice_scores = []
                for b in range(y.shape[0]):     # Batch items.
                    if mask[b, i]:
                        y_i = y[b, i]
                        y_hat_i = y_hat[b, i]
                        dice_score = dice(y_hat_i, y_i)
                        dice_scores.append(dice_score)

                if len(dice_scores) > 0:
                    mean_dice = np.mean(dice_scores)
                    self.log(f'val/dice/{region}', mean_dice, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

        # Log prediction images.
        if self.logger and self.current_epoch % self.__val_image_interval == 0:
            class_labels = {
                1: 'foreground'
            }
            if batch_idx < self.__max_image_batches:
                for i, desc in enumerate(descs):
                    # Plot for each channel.
                    for j in range(y.shape[1]):
                        # Skip channel if not present.
                        if not mask[i, j]:
                            continue

                        # Get images.
                        x_vol, y_vol, y_hat_vol = x[i, 0].cpu().numpy(), y[i, j], y_hat[i, j]

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
                            image = Image(
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
                            title = f'desc:{desc}:class:{j}:axis:{axis}'
                            self.logger.experiment.log({ title: image })
