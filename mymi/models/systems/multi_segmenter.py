import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import random
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.nn.functional as F
from typing import Dict, List, Literal, Optional, OrderedDict, Tuple, Union
from wandb import Image

from mymi import config
from mymi.geometry import get_extent_centre
from mymi import logging
from mymi.losses import DiceWithFocalLoss
from mymi.metrics import dice
from mymi.models import replace_ckpt_alias
from mymi.models.networks import MultiUNet3D
from mymi.types import PatientID, PatientRegions
from mymi.utils import arg_to_list, gpu_usage_nvml

LOG_ON_EPOCH = True
LOG_ON_STEP = False

class MultiSegmenter(pl.LightningModule):
    def __init__(
        self,
        complexity_weights_factor: float = 1,
        complexity_weights_window: int = 5,
        dw_cvg_delay_above: int = 20,
        dw_cvg_delay_below: int = 5,
        dw_cvg_thresholds: List[float] = [],
        dw_factor: float = 1,
        loss: nn.Module = DiceWithFocalLoss(),
        lr_init: float = 1e-3,
        metrics: List[str] = [],
        region: PatientRegions = None,
        use_complexity_weights: bool = False,
        use_downweighting: bool = False,
        use_lr_scheduler: bool = False,
        use_weights: bool = False,
        lr_milestones: List[int] = None,
        random_seed: float = 0,
        val_image_interval: int = 50,
        val_image_samples: Optional[List[PatientID]] = ['43', '44'],
        val_max_image_batches: Optional[int] = None,
        weight_decay: float = 0,
        weights: Optional[List[float]] = None,
        **kwargs):
        super().__init__()
        assert region is not None
        self.__complexity_weights_batch_losses = {}
        self.__complexity_weights_epoch_mean_losses = {}
        self.__complexity_weights_rolling_losses = {}
        self.__complexity_weights_factor = complexity_weights_factor
        self.__complexity_weights_window = complexity_weights_window
        self.__dw_batch_mean_dices = {}
        self.__dw_cvg_delay_above = dw_cvg_delay_above
        self.__dw_cvg_delay_below = dw_cvg_delay_below
        self.__dw_cvg_thresholds = dw_cvg_thresholds
        self.__dw_factor = dw_factor
        self.__loss = loss
        self.lr = lr_init        # 'self.lr' is default key that LR finder looks for on module.
        self.__name = None
        self.__metrics = metrics
        self.__random_seed = random_seed
        self.__regions = arg_to_list(region, str)
        self.__n_output_channels = len(self.__regions) + 1
        self.__network = MultiUNet3D(self.__n_output_channels, **kwargs)
        self.__use_complexity_weights = use_complexity_weights
        self.__use_downweighting = use_downweighting
        self.__use_lr_scheduler = use_lr_scheduler
        self.__use_weights = use_weights
        self.__lr_milestones = lr_milestones
        self.__val_image_interval = val_image_interval
        self.__val_image_samples = val_image_samples
        self.__val_max_image_batches = val_max_image_batches
        self.__weights = weights
        self.__weight_decay = weight_decay

        # Handle static weighting.
        if self.__use_weights:
            if len(self.__weights) != len(self.__regions):
                raise ValueError(f"Expected static weights '{self.__weights}' (len={len(self.__weights)}) to have same length as regions '{self.__regions}' (len={len(self.__regions)}).")
            logging.info(f"Using static weights '{self.__weights}'.")

        # Handle down-weighting.
        if self.__use_downweighting:
            if len(self.__dw_cvg_thresholds) != len(self.__regions):
                raise ValueError(f"Expected convergence thresholds '{self.__dw_cvg_thresholds}' (len={len(self.__dw_cvg_thresholds)}) to have same length as regions '{self.__regions}' (len={len(self.__regions)}).")
            logging.info(f"Using down-weighting with factor={self.__dw_factor}, thresholds={self.__dw_cvg_thresholds}, and delay=({self.__dw_cvg_delay_above},{self.__dw_cvg_delay_below}).")

            self.__dw_cvg_states = np.zeros(len(self.__regions), dtype=bool)
            self.__dw_cvg_epochs_above = np.empty(len(self.__regions))
            self.__dw_cvg_epochs_above[:] = np.nan
            self.__dw_cvg_epochs_below = np.empty(len(self.__regions))
            self.__dw_cvg_epochs_below[:] = np.nan

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
        check_epochs: Union[int, Literal[False]] = np.inf,
        **kwargs: Dict) -> pl.LightningModule:
        # Check that model training has finished.
        if check_epochs != False:
            filepath = os.path.join(config.directories.models, model_name, run_name, 'last.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            if state['epoch'] < check_epochs - 1:
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
            opt['lr_scheduler'] = MultiStepLR(self.__optimiser, self.__lr_milestones, gamma=0.1)
            # opt['lr_scheduler'] = ReduceLROnPlateau(self.__optimiser, factor=0.5, patience=200, verbose=True)

        return opt

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        return self.__network(x)

    def load_state_dict(self, state_dict, *args, **kwargs):
        # Load 'down-weighting' state.
        dw_state = state_dict.pop('down-weighting')
        self.__dw_cvg_epochs_above = dw_state['cvg-epochs-above']
        self.__dw_cvg_epochs_below = dw_state['cvg-epochs-below']
        self.__dw_cvg_states = dw_state['cvg-states']

        # Load random number generator state.
        rng_state = state_dict.pop('rng')
        np.random.set_state(rng_state['numpy'])
        random.setstate(rng_state['python'])
        torch.random.set_rng_state(rng_state['torch'])
        torch.cuda.random.set_rng_state(rng_state['torch-cuda'])

        super().load_state_dict(state_dict, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        # Add 'down-weighting' state.
        state_dict['down-weighting'] = {
            'cvg-epochs-above': self.__dw_cvg_epochs_above,
            'cvg-epochs-below': self.__dw_cvg_epochs_below,
            'cvg-states': self.__dw_cvg_states
        } 

        # Add random number generator state.
        state_dict['rng'] = {
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'torch': torch.random.get_rng_state(),
            'torch-cuda': torch.cuda.random.get_rng_state()
        }

        return state_dict

    def training_step(self, batch, batch_idx):
        # Set RNG at start of every training epoch.
        if batch_idx == 0:
            seed = self.__random_seed + self.current_epoch
            print(f"setting model seed {seed}")
            seed_everything(seed)

        # Forward pass.
        desc, x, y, mask, weights = batch
        if batch_idx < 5: 
            logging.info(f"Training... (epoch={self.current_epoch},batch={batch_idx},samples={desc})")

        batch_size = len(desc)

        # Apply static weights.
        if self.__use_weights:
            # Chain weights.
            weights = weights[0].cpu()

            # Apply static weighting.
            weights = [self.__weights[i - 1] * w if i != 0 else w for i, w in enumerate(weights)]
            weights = torch.Tensor([weights] * batch_size).to(x.device)

        # Apply down-weighting.
        if self.__use_downweighting:
            # Chain weights.
            weights = weights[0].cpu()

            # Apply down-weighting.
            weights = [(1 / (self.__dw_factor ** 2)) * w if i != 0 and self.__dw_cvg_states[i - 1] else w for i, w in enumerate(weights)]
            weights = torch.Tensor([weights] * batch_size).to(x.device)

        # Apply complexity weighting.
        if self.__use_complexity_weights:
            # Calculate distance from 'loss_min' per region.
            loss_min = -1
            rlosses = []
            for region in self.__regions:
                rloss = self.__complexity_weights_rolling_losses[region]
                rlosses.append(rloss)
            rlosses = np.array(rlosses)
            weights = rlosses - loss_min

            # Apply factor and normalise.
            weights = weights ** self.__complexity_weights_factor
            weights = list(weights / np.sum(weights))
            weights = [0] + weights
            weights = torch.Tensor([weights] * batch_size).to(x.device)

        # Normalise weights. Do this last as we might combine static and dynamic weights.
        weights = weights[0].cpu()
        weights = list(weights / weights.sum())
        weights = torch.Tensor([weights] * batch_size).to(x.device)

        # Log weights. First batch item only.
        for i, weight in enumerate(weights[0]):
            region = self.__channel_region_map[i]
            self.log(f'train/weight/{region}', weight, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

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
        if self.__use_complexity_weights:
            # Store loss per-region.
            loss = self.__loss(y_hat, y, mask, weights, reduce_channels=False)
            for region, loss in zip(self.__regions, loss[1:]):
                if region not in self.__complexity_weights_batch_losses:
                    self.__complexity_weights_batch_losses[region] = [loss.item()]
                else:
                    self.__complexity_weights_batch_losses[region] += [loss.item()]

            loss = loss.mean()
        else:
            loss = self.__loss(y_hat, y, mask, weights)

        self.log('val/loss', loss, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

        # Convert pred to binary mask.
        y = y.cpu().numpy()
        y_hat = F.one_hot(y_hat.argmax(axis=1), num_classes=y.shape[1]).moveaxis(-1, 1)  # 'F.one_hot' adds new axis last, move to second place.
        y_hat = y_hat.cpu().numpy().astype(bool)

        # Record gpu usage.
        for i, usage_mb in enumerate(gpu_usage_nvml()):
            self.log(f'gpu/{i}', usage_mb, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

        # Record dice.
        if 'dice' in self.__metrics:
            # Operate on each region separately.
            for i in range(self.__n_output_channels):
                # Skip 'background' channel.
                if i == 0:
                    continue

                # Calculate batch mean dice.
                region = self.__channel_region_map[i]
                dice_scores = []
                for b in range(y.shape[0]):
                    if mask[b, i]:
                        y_i = y[b, i]
                        y_hat_i = y_hat[b, i]
                        dice_score = dice(y_hat_i, y_i)
                        dice_scores.append(dice_score)
                if len(dice_scores) == 0:
                    # Skip if no dice scores for this region, for this batch (could have been masked out).
                    continue
                batch_mean_dice = np.mean(dice_scores)

                # Log to wandb.
                self.log(f'val/dice/{region}', batch_mean_dice, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

                # Save batch mean values for down-weighting convergence calculations.
                if self.__use_downweighting:
                    if region in self.__dw_batch_mean_dices:
                        self.__dw_batch_mean_dices[region] += [batch_mean_dice]
                    else:
                        self.__dw_batch_mean_dices[region] = [batch_mean_dice]
                        
        # Log prediction images.
        if self.logger:
            if self.current_epoch % self.__val_image_interval == 0 and (self.__val_max_image_batches is None or batch_idx < self.__val_max_image_batches):
                class_labels = {
                    1: 'foreground'
                }
                for i, desc in enumerate(descs):
                    if self.__val_image_samples is not None:
                        pat_id = desc.split(':')[1]
                        if pat_id not in self.__val_image_samples:
                            continue

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
                            region = self.__channel_region_map[j]
                            title = f'desc:{desc}:region:{region}:axis:{axis}'
                            self.logger.log_image(key=title, images=[image], step=self.global_step)

    def on_validation_epoch_end(self):
        if self.__use_complexity_weights:
            for i in range(self.__n_output_channels):
                # Skip 'background' channel.
                if i == 0:
                    continue

                # Calculate mean value.
                region = self.__channel_region_map[i]
                if region not in self.__complexity_weights_batch_losses:
                    # Skip if region wasn't present in validation samples.
                    logging.info(f"Skipping complexity weights for region '{region}'. Wasn't present in validation samples.")
                    continue
                epoch_mean_loss = np.mean(self.__complexity_weights_batch_losses[region])
                if region in self.__complexity_weights_epoch_mean_losses:
                    self.__complexity_weights_epoch_mean_losses[region] += [epoch_mean_loss]
                else:
                    self.__complexity_weights_epoch_mean_losses[region] = [epoch_mean_loss]
                self.log(f'train/cw-loss/{region}', epoch_mean_loss, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

                # Calculate rolling loss.
                epoch_mean_losses = self.__complexity_weights_epoch_mean_losses[region]
                if len(epoch_mean_losses) < self.__complexity_weights_window:
                    self.__complexity_weights_rolling_losses[region] = np.mean(epoch_mean_losses)
                else:
                    self.__complexity_weights_rolling_losses[region] = np.mean(epoch_mean_losses[-self.__complexity_weights_window:])
                self.log(f'train/cw-rolling-loss/{region}', self.__complexity_weights_rolling_losses[region], on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

            # Reset batch mean losses.
            self.__complexity_weights_batch_losses = {}

        if self.__use_downweighting:
            for i in range(self.__n_output_channels):
                # Skip 'background' channel.
                if i == 0:
                    continue

                # Calculate mean value.
                region = self.__channel_region_map[i]
                if region not in self.__dw_batch_mean_dices:
                    # Skip if region wasn't present in validation samples.
                    logging.info(f"Skipping down-weighting for region '{region}'. Wasn't present in validation samples.")
                    continue
                epoch_mean_dice = np.mean(self.__dw_batch_mean_dices[region])
                self.log(f'val/dw/dice/{region}', epoch_mean_dice, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

                # Check OAR convergence state.
                converged = self.__dw_cvg_states[i - 1]
                cvg_thresh = self.__dw_cvg_thresholds[i - 1]
                if not converged:
                    # Check if over convergence threshold.
                    if epoch_mean_dice >= cvg_thresh:
                        # Get epoch when first went over threshold.
                        epoch_above = self.__dw_cvg_epochs_above[i - 1] 
                        if np.isnan(epoch_above):
                            self.__dw_cvg_epochs_above[i - 1] = self.current_epoch
                        epoch_above = self.__dw_cvg_epochs_above[i - 1] 

                        # Check if region has converged.
                        epochs_above = self.current_epoch - epoch_above + 1
                        if epochs_above >= self.__dw_cvg_delay_above:
                            self.__dw_cvg_states[i - 1] = True
                            self.__dw_cvg_epochs_below[i - 1] = np.nan
                    else:
                        self.__dw_cvg_epochs_above[i - 1] = np.nan
                else:
                    if epoch_mean_dice < cvg_thresh:
                        # Get epoch when first went under threshold.
                        epoch_below = self.__dw_cvg_epochs_below[i - 1] 
                        if np.isnan(epoch_below):
                            self.__dw_cvg_epochs_below[i - 1] = self.current_epoch
                        epoch_below = self.__dw_cvg_epochs_below[i - 1] 

                        # Check if metric has unconverged.
                        epochs_below = self.current_epoch - epoch_below + 1
                        if epochs_below >= self.__dw_cvg_delay_below:
                            self.__dw_cvg_states[i - 1] = False
                            self.__dw_cvg_epochs_above[i - 1] = np.nan
                    else:
                        # Metric may have crossed back over threshold.
                        self.__dw_cvg_epochs_below[i - 1] = np.nan

                # Log convergence state.
                cvg = self.__dw_cvg_states[i - 1]
                self.log(f'val/dw/cvg/{region}', float(cvg), on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)
                thresh = self.__dw_cvg_thresholds[i - 1]
                self.log(f'val/dw/cvg/thresholds/{region}', thresh, on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)
                epoch_above = self.__dw_cvg_epochs_above[i - 1]
                if np.isnan(epoch_above):
                    epochs_above = 0
                else:
                    epochs_above = self.current_epoch - epoch_above + 1
                self.log(f'val/dw/cvg/epochs-above/{region}', float(epochs_above), on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)
                epoch_below = self.__dw_cvg_epochs_below[i - 1]
                if np.isnan(epoch_below):
                    epochs_below = 0
                else:
                    epochs_below = self.current_epoch - epoch_below + 1
                self.log(f'val/dw/cvg/epochs-below/{region}', float(epochs_below), on_epoch=LOG_ON_EPOCH, on_step=LOG_ON_STEP)

            # Reset batch means.
            self.__dw_batch_mean_dices = {}
