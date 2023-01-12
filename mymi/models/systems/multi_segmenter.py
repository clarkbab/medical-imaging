from monai.metrics import compute_dice
import numpy as np
import os
import pytorch_lightning as pl
from scipy.ndimage import center_of_mass
import torch
from torch import nn
from torch.optim import SGD
import torch.nn.functional as F
from typing import Dict, List, Optional, OrderedDict, Tuple
import wandb

from mymi import config
from mymi.geometry import get_extent_centre
from mymi import logging
from mymi.losses import DiceLoss
from mymi.metrics import dice
from mymi.models import replace_checkpoint_alias
from mymi.models.networks import MultiUNet3D
from mymi.postprocessing import get_batch_largest_cc
from mymi.regions import to_list
from mymi import types

class MultiSegmenter(pl.LightningModule):
    def __init__(
        self,
        regions: types.PatientRegions,
        loss: nn.Module,
        metrics: List[str] = [],
        pretrained_model: Optional[pl.LightningModule] = None,
        spacing: Optional[types.ImageSpacing3D] = None):
        super().__init__()
        if 'distances' in metrics and spacing is None:
            raise ValueError(f"Localiser requires 'spacing' when calculating 'distances' metric.")
        self.__distances_delay = 50
        self.__distances_interval = 20
        self.__loss = loss
        self.__max_image_batches = 5
        self.__name = None
        self.__metrics = metrics
        pretrained_model = pretrained_model.network if pretrained_model else None
        self.__spacing = spacing
        self.__regions = to_list(regions)
        self.__n_output_channels = len(to_list(self.__regions)) + 1
        self.__network = MultiUNet3D(self.__n_output_channels, pretrained_model=pretrained_model)

        # Create channel -> region map.
        self.__channel_region_map = { 0: 'background' }
        for i, region in enumerate(self.__regions):
            self.__channel_region_map[i + 1] = region 

    @property
    def network(self) -> nn.Module:
        return self.__network

    @property
    def name(self) -> Optional[Tuple[str, str, str]]:
        return self.__name

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
            n_samples = run_name.split('-')[-1]
            if n_samples == '5':
                n_epochs = 900
            elif n_samples == '10':
                n_epochs = 450
            elif n_samples == '20':
                n_epochs = 300
            else:
                n_epochs = 150
            if state['epoch'] < n_epochs - 1:
                raise ValueError(f"Can't load segmenter ('{model_name}','{run_name}','{checkpoint}') - hasn't completed {n_epochs} epochs training.")

        # Load model.
        model_name, run_name, checkpoint = replace_checkpoint_alias(model_name, run_name, checkpoint)
        filepath = os.path.join(config.directories.models, model_name, run_name, f"{checkpoint}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Segmenter '{model_name}' with run name '{run_name}' and checkpoint '{checkpoint}' not found.")

        # Update keys by adding '_Segmenter_' prefix if required.
        checkpoint_data = torch.load(filepath, map_location=torch.device('cpu'))
        pairs = []
        update = False
        for k, v in checkpoint_data['state_dict'].items():
            # Get new key.
            if not k.startswith('_Segmenter_'):
                update = True
                new_key = '_Segmenter_' + k
            else:
                new_key = k

            pairs.append((new_key, v))
        checkpoint_data['state_dict'] = OrderedDict(pairs)
        if update:
            logging.info(f"Updating checkpoint keys for model '{(model_name, run_name, checkpoint)}'.")
            torch.save(checkpoint_data, filepath)

        # Load checkpoint.
        segmenter = MultiSegmenter.load_from_checkpoint(filepath, **kwargs)
        segmenter.__name = (model_name, run_name, checkpoint)
        return segmenter

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(
        self,
        x: torch.Tensor,
        probs: bool = False):
        # Get prediction.
        pred = self.__network(x)
        if probs:
            return pred

        # Apply thresholding.
        pred = pred.argmax(dim=1)
        
        # Apply postprocessing.
        pred = pred.cpu().numpy().astype(np.bool_)
        pred = get_batch_largest_cc(pred)

        return pred

    def training_step(self, batch, _):
        # Forward pass.
        _, x, y, class_mask, class_weights = batch
        y_hat = self.__network(x)
        loss = self.__loss(y_hat, y, class_mask, class_weights)
        self.log('train/loss', loss, on_epoch=True)

        # Convert pred to binary mask.
        y = y.cpu().numpy()
        y_hat = F.one_hot(y_hat.argmax(axis=1), num_classes=y.shape[1]).moveaxis(-1, 1)  # 'F.one_hot' adds new axis last, move to second place.
        y_hat = y_hat.cpu().numpy().astype(bool)

        # Report metrics.
        if 'dice' in self.__metrics:
            # Accumulate dice scores for each element in batch.
            for i in range(self.__n_output_channels):     # Channels.
                region = self.__channel_region_map[i]
                dice_scores = []
                for b in range(y.shape[0]):     # Batch items.
                    if class_mask[b, i]:
                        y_i = y[b, i]   
                        y_hat_i = y_hat[b, i]
                        import os
                        filepath = '/tmp/tensors/pred.pt'
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        torch.save(y_hat_i, filepath)
                        filepath = '/tmp/tensors/label.pt'
                        torch.save(y_i, filepath)
                        print('saved tensors')
                        dice_score = dice(y_hat_i, y_i)
                        dice_scores.append(dice_score)

                channel_dice = np.mean(dice_scores)
                self.log(f'train/dice/{region}', channel_dice, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        _, x, y, class_mask, class_weights = batch
        y_hat = self.__network(x)
        loss = self.__loss(y_hat, y, class_mask, class_weights)
        self.log('val/loss', loss, on_epoch=True)

        # Convert pred to binary mask.
        y = y.cpu().numpy()
        y_hat = F.one_hot(y_hat.argmax(axis=1), num_classes=y.shape[1]).moveaxis(-1, 1)  # 'F.one_hot' adds new axis last, move to second place.
        y_hat = y_hat.cpu().numpy().astype(bool)

        if 'dice' in self.__metrics:
            # Accumulate dice scores for each element in batch.
            for i in range(self.__n_output_channels):     # Channels.
                region = self.__channel_region_map[i]
                dice_scores = []
                for b in range(y.shape[0]):     # Batch items.
                    if class_mask[b, i]:
                        y_i = y[b, i]
                        y_hat_i = y_hat[b, i]
                        import os
                        filepath = '/tmp/tensors/pred.pt'
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        torch.save(y_hat_i, filepath)
                        filepath = '/tmp/tensors/label.pt'
                        torch.save(y_i, filepath)
                        print('saved tensors')
                        dice_score = dice(y_hat_i, y_i)
                        dice_scores.append(dice_score)

                channel_dice = np.mean(dice_scores)
                self.log(f'train/dice/{region}', channel_dice, on_epoch=True)

        # Log predictions.
        # if self.logger:
        #     class_labels = {
        #         1: 'foreground'
        #     }

        #     for i, desc in enumerate(descs):
        #         if batch_idx > self.__max_image_batches + 1:
        #             break

        #         # Get images.
        #         x_vol, y_vol, y_hat_vol = x[i, 0].cpu().numpy(), y[i], y_hat[i]

        #         # Get centre of extent of ground truth.
        #         centre = get_extent_centre(y_vol)
        #         if centre is None:
        #             logging.info(f'Empty label, desc: {desc}. Sum: {y_vol.sum()}')
        #             continue
        #             # raise ValueError(f'Empty label, desc: {desc}. Sum: {y_vol.sum()}')

        #         for axis, centre_ax in enumerate(centre):
        #             # Get slices.
        #             slices = tuple([centre_ax if i == axis else slice(0, x_vol.shape[i]) for i in range(0, len(x_vol.shape))])
        #             x_img, y_img, y_hat_img = x_vol[slices], y_vol[slices], y_hat_vol[slices]

        #             # Fix orientation.
        #             if axis == 0 or axis == 1:
        #                 x_img = np.rot90(x_img)
        #                 y_img = np.rot90(y_img)
        #                 y_hat_img = np.rot90(y_hat_img)
        #             elif axis == 2:
        #                 x_img = np.transpose(x_img)
        #                 y_img = np.transpose(y_img) 
        #                 y_hat_img = np.transpose(y_hat_img)

        #             # Send image.
        #             image = wandb.Image(
        #                 x_img,
        #                 caption=desc,
        #                 masks={
        #                     'ground_truth': {
        #                         'mask_data': y_img,
        #                         'class_labels': class_labels
        #                     },
        #                     'predictions': {
        #                         'mask_data': y_hat_img,
        #                         'class_labels': class_labels
        #                     }
        #                 }
        #             )
        #             title = f'{desc}:axis:{axis}'
        #             self.logger.experiment.log({ title: image })
