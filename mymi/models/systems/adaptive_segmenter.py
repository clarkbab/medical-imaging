import csv
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import random
from scipy.ndimage import binary_dilation
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR, ReduceLROnPlateau
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
from mymi.types import ModelName, PatientID, PatientRegions
from mymi.utils import arg_to_list, gpu_usage_nvml

class AdaptiveSegmenter(pl.LightningModule):
    def __init__(
        self,
        complexity_weights_factor: float = 1,
        complexity_weights_window: int = 5,
        cw_cvg_calculate: bool = True,
        cw_cvg_delay_above: int = 20,
        cw_cvg_delay_below: int = 5,
        cw_cvg_thresholds: List[float] = [],
        cw_factor: Optional[Union[float, List[float]]] = None,
        cw_schedule: Optional[List[int]] = None,
        cyclic_min: Optional[float] = None,
        cyclic_max: Optional[float] = None,
        dilate_iters: Optional[List[int]] = None,
        dilate_region: Optional[PatientRegions] = None,
        dilate_schedule: Optional[List[int]] = None,
        log_on_epoch: bool = True,
        log_on_step: bool = False,
        loss: nn.Module = DiceWithFocalLoss(),
        lr_find: bool = False,
        lr_init: float = 1e-3,
        metrics: List[str] = [],
        model_name: str = 'model-name',
        region: PatientRegions = None,
        use_complexity_weights: bool = False,
        use_cvg_weighting: bool = False,
        use_dilation: bool = False,
        use_lr_scheduler: bool = False,
        use_weights: bool = False,
        lr_milestones: List[int] = None,
        random_seed: float = 0,
        run_name: str = 'run-name',
        val_image_interval: int = 50,
        val_image_samples: Optional[List[PatientID]] = None,
        val_max_image_batches: Optional[int] = None,
        weight_decay: float = 0,
        weights: Optional[Union[List[float], List[List[float]]]] = None,
        weights_schedule: Optional[List[int]] = None,
        **kwargs) -> None:
        super().__init__()
        assert region is not None
        self.__complexity_weights_batch_losses = {}
        self.__complexity_weights_epoch_mean_losses = {}
        self.__complexity_weights_rolling_losses = {}
        self.__complexity_weights_factor = complexity_weights_factor
        self.__complexity_weights_window = complexity_weights_window
        self.__cw_batch_mean_dices = {}
        self.__cw_cvg_calculate = cw_cvg_calculate
        self.__cw_cvg_delay_above = cw_cvg_delay_above
        self.__cw_cvg_delay_below = cw_cvg_delay_below
        self.__cw_cvg_thresholds = cw_cvg_thresholds
        self.__cw_factor = cw_factor
        self.__cw_schedule = cw_schedule
        self.__cyclic_min = cyclic_min
        self.__cyclic_max = cyclic_max
        self.__dilate_iters = dilate_iters
        self.__dilate_schedule = dilate_schedule
        self.__log_on_epoch = log_on_epoch
        self.__log_on_step = log_on_step
        self.__loss = loss
        self.lr = lr_init        # 'self.lr' is default key that LR finder looks for on module.
        self.__lr_find = lr_find
        self.__metrics = metrics
        self.__model_name = model_name
        self.__name = None
        self.__random_seed = random_seed
        self.__regions = arg_to_list(region, str)
        self.__run_name = run_name
        self.__n_input_channels = len(self.__regions) + 2
        self.__n_output_channels = len(self.__regions) + 1
        self.__network = MultiUNet3D(self.__n_output_channels, n_input_channels=self.__n_input_channels, **kwargs)
        self.__use_complexity_weights = use_complexity_weights
        self.__use_cvg_weighting = use_cvg_weighting
        self.__use_dilation = use_dilation
        self.__use_lr_scheduler = use_lr_scheduler
        self.__use_weights = use_weights
        self.__lr_milestones = lr_milestones
        self.__val_image_interval = val_image_interval
        self.__val_image_samples = val_image_samples
        self.__val_max_image_batches = val_max_image_batches
        self.__weights = weights
        self.__weights_schedule = weights_schedule
        self.__weight_decay = weight_decay

        # Handle label dilation.
        if self.__use_dilation:
            if len(self.__dilate_iters) != len(self.__dilate_schedule):
                raise ValueError(f"Expected 'dilate_iters' (len={len(self.__dilate_iters)}) to have same length as 'dilate_schedule' (len={len(self.__dilate_schedule)}).")

            dilate_regions = arg_to_list(dilate_region, str)
            if dilate_regions is None:
                self.__dilate_channels = list(range(1, len(self.__regions) + 1))
                dilate_regions = self.__regions
            else:
                self.__dilate_channels = []
                for i, region in enumerate(self.__regions):
                    if region in dilate_regions:
                        self.__dilate_channels.append(i + 1)

            logging.info(f"Using label dilation on regions '{dilate_regions}' (channels '{self.__dilate_channels}') with iters '{self.__dilate_iters}' and schedule '{self.__dilate_schedule}'.")

        # Handle weighting.
        if self.__use_weights:
            if self.__weights_schedule is not None:
                if len(self.__weights) != len(self.__weights_schedule):
                    raise ValueError(f"Expected weights (len={len(self.__weights)}) to have same length as schedule (len={len(self.__weights_schedule)}).")
                for weights in self.__weights:
                    if len(weights) != len(self.__regions):
                        raise ValueError(f"Expected weights '{weights}' (len={len(weights)}) to have same length as regions '{self.__regions}' (len={len(self.__regions)}).")
            else:
                if len(self.__weights) != len(self.__regions):
                    raise ValueError(f"Expected weights '{self.__weights}' (len={len(self.__weights)}) to have same length as regions '{self.__regions}' (len={len(self.__regions)}).")
            logging.info(f"Using weights '{self.__weights}' with schedule '{self.__weights_schedule}'.")

        # Create convergence tracking objects.
        # Do this even when 'use_cvg_weighting=False' as it allows us to track 
        # convergence via wandb API.
        if self.__cw_cvg_calculate:
            if len(self.__cw_cvg_thresholds) != len(self.__regions):
                raise ValueError(f"Expected convergence thresholds '{self.__cw_cvg_thresholds}' (len={len(self.__cw_cvg_thresholds)}) to have same length as regions '{self.__regions}' (len={len(self.__regions)}).")
            if self.__use_cvg_weighting:
                logging.info(f"Using convergence weighting with factor={self.__cw_factor}, schedule={self.__cw_schedule}, thresholds={self.__cw_cvg_thresholds}, and delay=({self.__cw_cvg_delay_above},{self.__cw_cvg_delay_below}).")

        self.__cw_cvg_states = np.zeros(len(self.__regions), dtype=bool)
        self.__cw_cvg_epochs_above = np.empty(len(self.__regions))
        self.__cw_cvg_epochs_above[:] = np.nan
        self.__cw_cvg_epochs_below = np.empty(len(self.__regions))
        self.__cw_cvg_epochs_below[:] = np.nan

        # Create channel -> region map.
        self.__channel_region_map = { 0: 'background' }
        for i, region in enumerate(self.__regions):
            self.__channel_region_map[i + 1] = region 

        if self.__lr_find:
            # Create CSV file.
            filepath = os.path.join(config.directories.lr_find, self.__model_name, self.__run_name, 'lr-find.csv')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', newline='') as f:
                csv_writer = csv.writer(f)
                header = ['region', 'step', 'loss']
                csv_writer.writerow(header)

    @property
    def name(self) -> Optional[Tuple[str, str, str]]:
        return self.__name

    @property
    def network(self) -> nn.Module:
        return self.__network

    @staticmethod
    def load(
        model: ModelName,
        check_epochs: bool = True,
        cw_cvg_calculate: bool = False,
        n_epochs: Optional[int] = np.inf,
        **kwargs: Dict) -> pl.LightningModule:
        # Check that model training has finished.
        if check_epochs:
            last_model = replace_ckpt_alias((model[0], model[1], 'last'))
            filepath = os.path.join(config.directories.models, last_model[0], last_model[1], f'{last_model[2]}.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            n_epochs_complete = state['epoch'] + 1
            if n_epochs_complete < n_epochs:
                raise ValueError(f"Can't load multi-segmenter '{model}', has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model = replace_ckpt_alias(model)
        filepath = os.path.join(config.directories.models, model[0], model[1], f"{model[2]}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Multi-segmenter '{model}' not found.")

        # Update keys by adding '_Segmenter_' prefix if required.
        ckpt_data = torch.load(filepath, map_location=torch.device('cpu'))
        pairs = []
        update = False
        for k, v in ckpt_data['state_dict'].items():
            # Get new key.
            if k.startswith('_Segmenter__AdaptiveSegmenter__network'):
                update = True
                new_k = k.replace('_Segmenter__AdaptiveSegmenter__network', '_AdaptiveSegmenter__network')
            else:
                new_k = k
            pairs.append((new_k, v))
        ckpt_data['state_dict'] = OrderedDict(pairs)
        if update:
            logging.info(f"Updating checkpoint keys for model '{model}'.")
            torch.save(ckpt_data, filepath)

        # Load checkpoint.
        segmenter = AdaptiveSegmenter.load_from_checkpoint(filepath, cw_cvg_calculate=cw_cvg_calculate, **kwargs)
        segmenter.__name = model
        return segmenter

    def configure_optimizers(self):
        self.__optimiser = Adam(self.parameters(), lr=self.lr, weight_decay=self.__weight_decay) 
        opt = {
            'optimizer': self.__optimiser,
            'monitor': 'val/loss'
        }
        if self.__use_lr_scheduler:
            if self.__cyclic_min is None or self.__cyclic_max is None:
                raise ValueError(f"Both 'cyclic_min', and 'cyclic_max' must be specified when using cyclic LR.")

            opt['lr_scheduler'] = CyclicLR(self.__optimiser, self.__cyclic_min, self.__cyclic_max)
            # opt['lr_scheduler'] = MultiStepLR(self.__optimiser, self.__lr_milestones, gamma=0.1)
            # opt['lr_scheduler'] = ReduceLROnPlateau(self.__optimiser, factor=0.5, patience=200, verbose=True)

            logging.info(f"Using cyclic LR with min={self.__cyclic_min}, max={self.__cyclic_max}).")

        return opt

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        return self.__network(x)

    def load_state_dict(self, state_dict, *args, **kwargs):
        if 'down-weighting' in state_dict:
            cw_state = state_dict.pop('down-weighting')
            self.__cw_cvg_epochs_above = cw_state['cvg-epochs-above']
            self.__cw_cvg_epochs_below = cw_state['cvg-epochs-below']
            self.__cw_cvg_states = cw_state['cvg-states']

        # # Load random number generator state.
        rng_state = state_dict.pop('rng')
        # np.random.set_state(rng_state['numpy'])
        # random.setstate(rng_state['python'])
        # torch.random.set_rng_state(rng_state['torch'])
        # torch.cuda.random.set_rng_state(rng_state['torch-cuda'])

        super().load_state_dict(state_dict, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        state_dict['down-weighting'] = {
            'cvg-epochs-above': self.__cw_cvg_epochs_above,
            'cvg-epochs-below': self.__cw_cvg_epochs_below,
            'cvg-states': self.__cw_cvg_states
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
        # Forward pass.
        desc, x, y, mask, weights = batch
        if batch_idx < 5: 
            pass
            # logging.info(f"Training... (epoch={self.current_epoch},batch={batch_idx},samples={desc})")

        batch_size = len(desc)

        # Handle label dilation.
        if self.__use_dilation:
            # Determine current 'n_iter'.
            n_iter_curr = None
            for epoch, n_iter in zip(self.__dilate_schedule, self.__dilate_iters):
                if self.current_epoch >= epoch:
                    n_iter_curr = n_iter
                else:
                    break
            assert n_iter_curr is not None

            self.log(f'train/dilation/n_iter', float(n_iter_curr), on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
                
            # Apply dilation to each channel.
            for i, region in enumerate(self.__regions):
                channel = i + 1
                if channel in self.__dilate_channels:
                    dilating = 1
                    y_c = y[:, channel]
                    y_c_bs = []
                    for y_c_b in y_c:
                        y_c_b = binary_dilation(y_c_b.cpu().numpy(), iterations=n_iter_curr)
                        y_c_b = torch.Tensor(y_c_b).to(x.device)
                        y_c_bs.append(y_c_b)
                    y_c = torch.stack(y_c_bs)
                    y[:, channel] = y_c
                else:
                    dilating = 0

                self.log(f'train/dilation/{region}', float(dilating), on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Apply weights.
        if self.__use_weights:
            # Chain weights.
            weights = weights[0].cpu()

            # Get active weights.
            if self.__weights_schedule is None:
                static_weights = self.__weights
            else:
                schedule_i = np.max(np.where(np.array(self.__weights_schedule) <= self.current_epoch))
                static_weights = self.__weights[schedule_i]

            # Apply weighting.
            weights = [static_weights[i - 1] * w if i != 0 else w for i, w in enumerate(weights)]
            weights = torch.Tensor([weights] * batch_size).to(x.device)

        # Apply down-weighting.
        if self.__use_cvg_weighting:
            # Chain weights.
            weights = weights[0].cpu()

            # Get active weights.
            if self.__weights_schedule is None:
                cw_factor = self.__cw_factor
            else:
                schedule_i = np.max(np.where(np.array(self.__weights_schedule) <= self.current_epoch))
                cw_factor = self.__cw_factor[schedule_i]

            # Apply down-weighting.
            weights = [(cw_factor + 1) * w if i != 0 and not self.__cw_cvg_states[i - 1] else w for i, w in enumerate(weights)]
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
            self.log(f'train/weight/{region}', weight, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        y_hat = self.forward(x)
        include_background = False
        reduce_channels = False
        reduction = 'mean'
        loss = self.__loss(y_hat, y, include_background=include_background, mask=mask, weights=weights, reduce_channels=reduce_channels, reduction=reduction)
        region_losses = {}
        if not reduce_channels:
            # Log OAR loss.
            for i, l in enumerate(loss):
                if not torch.isnan(l).any():
                    region = self.__channel_region_map[i + 1]
                    self.log(f'train/loss/region/{region}', l, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

                    region_losses[region] = l

                    # if self.__lr_find:
                    #     self.__write_loss(region, self.global_step, l.item())

            # Reduce channels.
            if reduction == 'mean':
                loss = loss.mean()
            elif reduction == 'sum':
                loss = loss.sum()

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
            self.log('train/loss', loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

            # if self.__lr_find:
            #     self.__write_loss('all', self.global_step, loss.item())

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
                    self.log(f'train/dice/{region}', mean_dice, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Add main loss.
        losses = {}
        losses['loss'] = loss

        # Add region losses - for LR find.
        if not reduce_channels:
            losses.update(region_losses)

        return losses

    def validation_step(self, batch, batch_idx):
        # Forward pass.
        descs, x, y, mask, weights = batch
        y_hat = self.forward(x)
        include_background = False
        if self.__use_complexity_weights:
            # Store loss per-region.
            loss = self.__loss(y_hat, y, include_background=include_background, mask=mask, weights=weights, reduce_channels=False)
            for region, loss in zip(self.__regions, loss[1:]):
                if region not in self.__complexity_weights_batch_losses:
                    self.__complexity_weights_batch_losses[region] = [loss.item()]
                else:
                    self.__complexity_weights_batch_losses[region] += [loss.item()]

            loss = loss.mean()
        else:
            loss = self.__loss(y_hat, y, include_background=include_background, mask=mask, weights=weights)

        self.log('val/loss', loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Convert pred to binary mask.
        y = y.cpu().numpy()
        y_hat = F.one_hot(y_hat.argmax(axis=1), num_classes=y.shape[1]).moveaxis(-1, 1)  # 'F.one_hot' adds new axis last, move to second place.
        y_hat = y_hat.cpu().numpy().astype(bool)

        # Record gpu usage.
        for i, usage_mb in enumerate(gpu_usage_nvml()):
            self.log(f'gpu/{i}', usage_mb, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

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
                self.log(f'val/dice/{region}', batch_mean_dice, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

                # Save batch mean values for "convergence weighting" calculations.
                # Do this even when 'use_cvg_weighting=False' as it allows us to track 
                # convergence via wandb API.
                if self.__cw_cvg_calculate:
                    if region in self.__cw_batch_mean_dices:
                        self.__cw_batch_mean_dices[region] += [batch_mean_dice]
                    else:
                        self.__cw_batch_mean_dices[region] = [batch_mean_dice]
                        
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
                        if centre is None:
                            # Presumably data augmentation has pushed the label out of view.
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
                            region = self.__channel_region_map[j]
                            title = f'desc:{desc}:region:{region}:axis:{axis}'
                            caption = desc,
                            masks = {
                                'ground_truth': {
                                    'mask_data': y_img,
                                    'class_labels': class_labels
                                },
                                'predictions': {
                                    'mask_data': y_hat_img,
                                    'class_labels': class_labels
                                }
                            }
                            self.logger.log_image(key=title, images=[x_img], caption=caption, masks=[masks], step=self.global_step)

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
                self.log(f'train/cw-loss/{region}', epoch_mean_loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

                # Calculate rolling loss.
                epoch_mean_losses = self.__complexity_weights_epoch_mean_losses[region]
                if len(epoch_mean_losses) < self.__complexity_weights_window:
                    self.__complexity_weights_rolling_losses[region] = np.mean(epoch_mean_losses)
                else:
                    self.__complexity_weights_rolling_losses[region] = np.mean(epoch_mean_losses[-self.__complexity_weights_window:])
                self.log(f'train/cw-rolling-loss/{region}', self.__complexity_weights_rolling_losses[region], on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

            # Reset batch mean losses.
            self.__complexity_weights_batch_losses = {}

        # Calculate convergence state.
        # Do this even when 'use_cvg_weighting=False' as it allows us to track 
        # convergence via wandb API.
        if self.__cw_cvg_calculate:
            for i in range(self.__n_output_channels):
                # Skip 'background' channel.
                if i == 0:
                    continue

                # Calculate mean value.
                region = self.__channel_region_map[i]
                if region not in self.__cw_batch_mean_dices:
                    # Skip if region wasn't present in validation samples.
                    logging.info(f"Skipping \"convergence weighting\" for region '{region}'. Wasn't present in validation samples.")
                    continue
                epoch_mean_dice = np.mean(self.__cw_batch_mean_dices[region])
                self.log(f'val/dw/dice/{region}', epoch_mean_dice, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

                # Check OAR convergence state.
                converged = self.__cw_cvg_states[i - 1]
                cvg_thresh = self.__cw_cvg_thresholds[i - 1]
                if not converged:
                    # Check if over convergence threshold.
                    if epoch_mean_dice >= cvg_thresh:
                        # Get epoch when first went over threshold.
                        epoch_above = self.__cw_cvg_epochs_above[i - 1] 
                        if np.isnan(epoch_above):
                            self.__cw_cvg_epochs_above[i - 1] = self.current_epoch
                        epoch_above = self.__cw_cvg_epochs_above[i - 1] 

                        # Check if region has converged.
                        epochs_above = self.current_epoch - epoch_above + 1
                        if epochs_above >= self.__cw_cvg_delay_above:
                            self.__cw_cvg_states[i - 1] = True
                            self.__cw_cvg_epochs_below[i - 1] = np.nan
                    else:
                        self.__cw_cvg_epochs_above[i - 1] = np.nan
                else:
                    if epoch_mean_dice < cvg_thresh:
                        # Get epoch when first went under threshold.
                        epoch_below = self.__cw_cvg_epochs_below[i - 1] 
                        if np.isnan(epoch_below):
                            self.__cw_cvg_epochs_below[i - 1] = self.current_epoch
                        epoch_below = self.__cw_cvg_epochs_below[i - 1] 

                        # Check if metric has unconverged.
                        epochs_below = self.current_epoch - epoch_below + 1
                        if epochs_below >= self.__cw_cvg_delay_below:
                            self.__cw_cvg_states[i - 1] = False
                            self.__cw_cvg_epochs_above[i - 1] = np.nan
                    else:
                        # Metric may have crossed back over threshold.
                        self.__cw_cvg_epochs_below[i - 1] = np.nan

                # Log convergence state.
                cvg = self.__cw_cvg_states[i - 1]
                self.log(f'val/dw/cvg/{region}', float(cvg), on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
                thresh = self.__cw_cvg_thresholds[i - 1]
                self.log(f'val/dw/cvg/thresholds/{region}', thresh, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
                epoch_above = self.__cw_cvg_epochs_above[i - 1]
                if np.isnan(epoch_above):
                    epochs_above = 0
                else:
                    epochs_above = self.current_epoch - epoch_above + 1
                self.log(f'val/dw/cvg/epochs-above/{region}', float(epochs_above), on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
                epoch_below = self.__cw_cvg_epochs_below[i - 1]
                if np.isnan(epoch_below):
                    epochs_below = 0
                else:
                    epochs_below = self.current_epoch - epoch_below + 1
                self.log(f'val/dw/cvg/epochs-below/{region}', float(epochs_below), on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

            # Reset batch means.
            self.__cw_batch_mean_dices = {}

    def __write_loss(
        self,
        region: str,
        step: int,
        loss: float) -> None:
        # Write to CSV file.
        data = [region, step, loss]
        filepath = os.path.join(config.directories.lr_find, self.__model_name, self.__run_name, 'lr-find.csv')
        with open(filepath, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(data)
