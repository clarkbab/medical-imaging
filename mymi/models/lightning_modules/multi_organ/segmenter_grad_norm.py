import numpy as np
import os
import pytorch_lightning as pl
import torch
from typing import *

from mymi import config
from mymi.geometry import centre_of_extent
from mymi import logging
from mymi.losses import *
from mymi.metrics import *
from mymi.typing import *
from mymi.utils import *

from ...architectures import create_mednext_v1, layer_summary, UNet3D
from ...models import replace_ckpt_alias

class SegmenterGradNorm(pl.LightningModule):
    def __init__(
        self,
        regions: Regions,
        arch: str = 'unet3d:m',
        gn_alpha: float = 1.0,
        gn_enabled: bool = True,
        gn_loss_fn: Literal['abs', 'squared'] = 'abs',
        gn_lr_init: float = 1e-3,
        gn_clip_mult: Optional[float] = None,
        gn_softmax: bool = False,
        loss_fn: str = 'dice',
        loss_smoothing: float = 0.1,
        lr_init: float = 1e-4,
        name: Optional[ModelName] = None,
        save_training_metrics: bool = False,
        loss_record_interval: TrainingInterval = 'step:1',     # Loss should typically be recorded more frequently - it's not very expensive.
        metrics_record_interval: TrainingInterval = 'step:1',
        metrics_save_interval: TrainingInterval = 'epoch:end',
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        val_image_interval: int = 50,
        val_loss_record_interval: TrainingInterval = 'epoch:1',   # Requires its own parameter as step could have any value at the end of the epoch.
        val_max_image_batches: Optional[int] = 3,
        val_metrics_record_interval: TrainingInterval = 'epoch:1',   # Requires own parameter.
        wandb_log_on_epoch: bool = True,    # If True, regardless of our logging frequency, wandb will accumulate values over the epoch before logging.
        wandb_loss_record_interval: TrainingInterval = 'step:1',
        wandb_metrics_record_interval: TrainingInterval = 'step:5',
        wandb_val_loss_record_interval: TrainingInterval = 'epoch:1',
        wandb_val_metrics_record_interval: TrainingInterval = 'epoch:1',    # Requires own parameter.
        **kwargs) -> None:
        super().__init__()
        self.__metrics_record_interval = metrics_record_interval
        self.__loss_record_interval = loss_record_interval
        self.__val_loss_record_interval = val_loss_record_interval
        self.__val_metrics_record_interval = val_metrics_record_interval
        self.__metrics_save_interval = metrics_save_interval
        self.__wandb_log_on_epoch = wandb_log_on_epoch
        if loss_fn == 'dice':
            logging.info(f"Using DiceLoss with smoothing={loss_smoothing}.")
            self.__loss_fn = DiceLoss(smoothing=loss_smoothing)
        elif loss_fn == 'dml1':
            logging.info(f"Using DML1Loss with smoothing={loss_smoothing}.")
            self.__loss_fn = DML1Loss(smoothing=loss_smoothing)
        elif loss_fn == 'dml2':
            logging.info(f"Using DML2Loss with smoothing={loss_smoothing}.")
            self.__loss_fn = DML2Loss(smoothing=loss_smoothing)
        elif loss_fn == 'tversky':
            logging.info(f"Using TverskyLoss with alpha={tversky_alpha}, beta={tversky_beta}, smoothing={loss_smoothing}.")
            self.__loss_fn = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smoothing=loss_smoothing)
        elif loss_fn == 'tversky-dist':
            logging.info(f"Using TverskyDistanc3Loss with alpha={tversky_alpha}, beta={tversky_beta}, smoothing={loss_smoothing}.")
            self.__loss_fn = TverskyDistanceLoss(alpha=tversky_alpha, beta=tversky_beta, smoothing=loss_smoothing)
        else:
            raise ValueError(f"Unknown loss function '{loss_fn}'.")
        self.lr = lr_init        # 'self.lr' is default key that LR finder looks for on module.
        self.__name = name
        self.__regions = regions
        self.__n_channels = len(self.__regions) + 1
        self.__gn_alpha = gn_alpha
        self.__gn_enabled = gn_enabled
        if gn_loss_fn == 'abs':
            self.__gn_loss_fn = torch.abs
        elif gn_loss_fn == 'squared':
            self.__gn_loss_fn = lambda x: x ** 2
        else:
            raise ValueError(f"Unknown GradNorm loss function '{gn_loss_fn}'.")
        # These weights will be updated by a separate optimiser - perhaps with a different learning rate.
        self.__gn_task_weights = torch.nn.Parameter(torch.ones(len(self.__regions), dtype=torch.float32))
        self.__gn_init_task_losses = None
        self.__gn_lr_init = gn_lr_init
        self.__gn_clip_mult = gn_clip_mult
        self.__gn_softmax = gn_softmax
        self.__hooks_enabled = True
        if arch == 'unet3d:m':
            logging.info(f"Using UNet3D (M) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, **kwargs)
            self.__leaf_modules = layer_summary(arch, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
            self.__gn_balance_point = list(self.__network.layers[62].layer.parameters())
            self.__gn_clipping_layer = '_UNet3D__layers.61._LayerWrapper__layer'   # Output of final ReLU, before task-splitting using 1x1x1 conv.
            self.__gn_clipping_accum_layer = '_UNet3D__layers.62._LayerWrapper__layer'
        elif arch == 'unet3d:l':
            logging.info(f"Using UNet3D (L) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, n_features=64, **kwargs)
            self.__leaf_modules = layer_summary(arch, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
            self.__gn_balance_point = list(self.__network.layers[62].layer.parameters())
            self.__gn_clipping_layer = '_UNet3D__layers.61._LayerWrapper__layer'   # Output of final ReLU, before task-splitting using 1x1x1 conv.
            self.__gn_clipping_accum_layer = '_UNet3D__layers.62._LayerWrapper__layer'
        elif arch == 'unet3d:xl':
            logging.info(f"Using UNet3D (XL) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, n_features=128, **kwargs)
            self.__leaf_modules = layer_summary(arch, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
            self.__gn_balance_point = list(self.__network.layers[62].layer.parameters())
            self.__gn_clipping_layer = '_UNet3D__layers.61._LayerWrapper__layer'   # Output of final ReLU, before task-splitting using 1x1x1 conv.
            self.__gn_clipping_accum_layer = '_UNet3D__layers.62._LayerWrapper__layer'
        elif arch == 'mednext:s':
            logging.info(f"Using MedNeXt (S) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'S')
            self.__leaf_modules = layer_summary(arch, 1, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, 1, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
        elif arch == 'mednext:b':
            logging.info(f"Using MedNeXt (B) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'B')
            self.__leaf_modules = layer_summary(arch, 1, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, 1, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
        elif arch == 'mednext:m':
            logging.info(f"Using MedNeXt (M) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'M')
            self.__leaf_modules = layer_summary(arch, 1, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, 1, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
        elif arch == 'mednext:l':
            logging.info(f"Using MedNeXt (L) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'L')
            self.__leaf_modules = layer_summary(arch, 1, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, 1, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
        else:
            raise ValueError(f"Unknown architecture '{arch}'.")
        self.__save_training_metrics = save_training_metrics
        self.__wandb_loss_record_interval = wandb_loss_record_interval
        self.__wandb_metrics_record_interval = wandb_metrics_record_interval
        self.__val_image_interval = val_image_interval
        self.__val_max_image_batches = val_max_image_batches
        self.__wandb_val_metrics_record_interval = wandb_val_metrics_record_interval
        self.__wandb_val_loss_record_interval = wandb_val_loss_record_interval

        if self.__save_training_metrics:
            logging.info("Saving training metrics.")
            self.__register_training_metrics_hooks()

        if self.__gn_clip_mult is not None:
            self.__register_clipping_hook()

    @property
    def name(self) -> ModelName:
        return self.__name

    @property
    def network(self) -> torch.nn.Module:
        return self.__network

    @staticmethod
    def load(
        model: ModelName,
        check_epochs: bool = True,
        n_epochs: Optional[int] = np.inf,
        **kwargs: Dict) -> pl.LightningModule:
        # Check that model training has finished.
        if check_epochs:
            last_model = replace_ckpt_alias((model[0], model[1], 'last'))
            filepath = os.path.join(config.directories.models, last_model[0], last_model[1], f'{last_model[2]}.ckpt')
            state = torch.load(filepath, map_location=torch.device('cpu'))
            n_epochs_complete = state['epoch'] + 1
            if n_epochs_complete < n_epochs:
                raise ValueError(f"Can't load SegmenterGradNorm '{model}', has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model = replace_ckpt_alias(model)
        filepath = os.path.join(config.directories.models, model[0], model[1], f"{model[2]}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"SegmenterGradNorm '{model}' not found.")

        # Load checkpoint.
        segmenter = SegmenterGradNorm.load_from_checkpoint(filepath, name=model, **kwargs)

        return segmenter

    def configure_optimizers(self):
        # self.__optimiser = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        # This optimiser shouldn't touch task weight gradients - it would just set everything to zero :)
        self.__optimiser = torch.optim.AdamW(self.__network.parameters(), lr=self.lr) 
        self.__task_optimiser = torch.optim.AdamW([self.__gn_task_weights], lr=self.__gn_lr_init)
        logging.info(f"Using optimiser '{self.__optimiser}' with learning rate '{self.lr}'.")
        logging.info(f"Using task optimiser '{self.__task_optimiser}' with learning rate '{self.__gn_lr_init}'.")

        opt = {
            'optimizer': self.__optimiser,
            'monitor': 'val/loss'
        }
        opt['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimiser, factor=0.8, mode='min', patience=100, verbose=True)
        logging.info(f"Using optimiser scheduler 'ReduceLROnPlateau' with factor '0.8' and patience '100'.")

        return opt

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        return self.__network(x)

    def training_step(self, batch, _):
        desc, x, y, mask = batch
        n_batch_items = len(desc)

        # Forward pass.
        y_hat = self.forward(x)

        # Loss calculation.
        kwargs = dict(
            mask=mask,
            include_background=False,
            reduce_channels=False,
        )
        unweighted_task_losses = self.__loss_fn(y_hat, y, **kwargs) 

        # # Store initial task losses for GradNorm.
        # if self.__gn_init_task_losses is None:
        #     self.__gn_init_task_losses = loss.detach()
        # A better initial task loss might be our theoretical value, i.e. 1 for DiceLoss.
        if self.__gn_init_task_losses is None:
            self.__gn_init_task_losses = torch.ones_like(unweighted_task_losses, dtype=torch.float32)

        # Calculate loss ratios - will be less than 1.
        ls = (unweighted_task_losses / self.__gn_init_task_losses)

        for r, ri in zip(self.__regions, ls):
            if self.logger is not None:
                self.log(f'train/gn-l/{r}', ri, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'gradnorm-l-{r}',
                    'value': ri.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

        # Calculate inverse training rates - can be greater than 1 and hence GradNorm will push
        # slow-learning task gradients above the mean gradient.
        self.__rs = (ls / ls.mean())

        for r, ri in zip(self.__regions, self.__rs):
            if self.logger is not None:
                self.log(f'train/gn-r/{r}', ri, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'gradnorm-r-{r}',
                    'value': ri.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

        # Calculate task-specific losses.
        # Run weights through softmax before applying to losses. This allows negative weights for tasks, without
        # starting to maximise task-specific loss components. This might help during optimisation, allowing larger
        # learning rates for Gradnorm optimiser without going negative.

        if self.__gn_softmax:
            weights = len(self.__gn_task_weights) * torch.nn.functional.softmax(self.__gn_task_weights, dim=0)
        else:
            weights = self.__gn_task_weights
        task_losses = weights * unweighted_task_losses

        for r, w in zip(self.__regions, weights):
            if self.logger is not None:
                # Log task weights.
                self.log(f'train/gn-weights/{r}', w, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'gradnorm-weight-{r}',
                    'value': w.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

        if self.__gn_softmax:
            for r, w in zip(self.__regions, self.__gn_task_weights):
                if self.logger is not None:
                    # Log task weights.
                    self.log(f'train/gn-weights-task/{r}', w, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': 'train',
                        'module': '',
                        'module-type': '',
                        'shape': '',
                        'metric': f'gradnorm-weight-task-{r}',
                        'value': w.item(),
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

        for r, l in zip(self.__regions, task_losses):
            if self.logger is not None:
                self.log(f'train/loss/{r}', l, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'loss-{r}',
                    'value': l.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

        # Log learning rate as the default logger doesn't log against the epoch.
        self.log('train/lr', self.lr, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Report metrics.
        if (self.logger is not None and self.__interval_matches(self.__wandb_metrics_record_interval)) or \
            (self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval)):
                # Convert from softmax to binary mask.
                y_hat = torch.nn.functional.one_hot(y_hat.argmax(axis=1), num_classes=self.__n_channels)
                y_hat = y_hat.moveaxis(-1, 1)
                y_hat = y_hat.cpu().numpy().astype(bool)
                y = y.cpu().numpy()

                # Save masks for TP/TN/FP/FN gradient calculations - during backward hook
                # that doesn't have access to preds and labels.
                if self.__save_training_metrics:
                    self.__tp_mask = torch.tensor((y_hat == 1) & (y == 1), dtype=torch.bool, device=x.device)
                    self.__tn_mask = torch.tensor((y_hat == 0) & (y == 0), dtype=torch.bool, device=x.device)
                    self.__fp_mask = torch.tensor((y_hat == 1) & (y == 0), dtype=torch.bool, device=x.device)
                    self.__fn_mask = torch.tensor((y_hat == 0) & (y == 1), dtype=torch.bool, device=x.device)

                # Calculate mean stats over batch items.
                for i, r in enumerate(self.__regions):
                    c = i + 1
                    dices = []
                    tps = []
                    fps = []
                    fns = []
                    tns = []
                    for b in range(n_batch_items):
                        if mask[b, c]:
                            # Get batch item.
                            y_hat_c = y_hat[b, c]
                            y_c = y[b, c]   

                            # Calculate dice.
                            d = dice(y_hat_c, y_c)
                            dices.append(d)

                            # Calculate classification stats.
                            tp = true_positive_rate(y_hat_c, y_c)
                            fp = false_positive_rate(y_hat_c, y_c)
                            fn = false_negative_rate(y_hat_c, y_c)
                            tn = true_negative_rate(y_hat_c, y_c)
                            tps.append(tp)
                            fps.append(fp)
                            fns.append(fn)
                            tns.append(tn)

                    metrics = ['dice', 'tp', 'fp', 'fn', 'tn']
                    values = [dices, tps, fps, fns, tns]
                    for m, v in zip(metrics, values):
                        if len(v) > 0:
                            mean_value = np.mean(v)
                            if self.logger is not None and self.__interval_matches(self.__wandb_metrics_record_interval):
                                self.log(f'train/{m}/{r}', mean_value, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                                data = {
                                    'epoch': self.current_epoch,
                                    'step': self.global_step,
                                    'mode': 'train',
                                    'module': '',
                                    'module-type': '',
                                    'shape': '',
                                    'metric': f'{m}-{r}',
                                    'value': mean_value,
                                }
                                self.__training_metrics = append_row(self.__training_metrics, data)

        return task_losses

    def validation_step(self, batch, batch_idx):
        descs, x, y, mask = batch
        n_batch_items = len(descs)

        # Forward pass.
        y_hat = self.forward(x)

        # Loss calculation.
        kwargs = dict(
            mask=mask,
            include_background=False,
            reduce_channels=False,
        )
        task_losses = self.__loss_fn(y_hat, y, **kwargs) 

        # Log region losses.
        if self.logger is not None:
            for r, l in zip(self.__regions, task_losses):
                self.log(f'val/loss/{r}', l, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Record gpu usage.
        for i, usage_mb in enumerate(gpu_usage_nvml()):
            self.log(f'gpu/{i}', usage_mb, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Report metrics.
        if (self.logger is not None and self.__interval_matches(self.__wandb_val_metrics_record_interval)) or \
            self.__interval_matches(self.__val_metrics_record_interval):
                # Convert from softmax to binary mask.
                y_hat = torch.nn.functional.one_hot(y_hat.argmax(axis=1), num_classes=self.__n_channels)
                y_hat = y_hat.moveaxis(-1, 1)
                y_hat = y_hat.cpu().numpy().astype(bool)
                y = y.cpu().numpy()

                # Calculate mean stats over batch items.
                for i, r in enumerate(self.__regions):
                    c = i + 1
                    dices = []
                    tps = []
                    fps = []
                    fns = []
                    tns = []
                    for b in range(n_batch_items):
                        if mask[b, c]:
                            # Get batch item.
                            y_hat_c = y_hat[b, c]
                            y_c = y[b, c]   

                            # Calculate dice.
                            d = dice(y_hat_c, y_c)
                            dices.append(d)

                            # Calculate classification stats.
                            tp = true_positive_rate(y_hat_c, y_c)
                            fp = false_positive_rate(y_hat_c, y_c)
                            fn = false_negative_rate(y_hat_c, y_c)
                            tn = true_negative_rate(y_hat_c, y_c)
                            tps.append(tp)
                            fps.append(fp)
                            fns.append(fn)
                            tns.append(tn)

                    metrics = ['dice', 'tp', 'fp', 'fn', 'tn']
                    values = [dices, tps, fps, fns, tns]
                    for m, v in zip(metrics, values):
                        if len(v) > 0:
                            mean_value = np.mean(v)
                            if self.logger is not None and self.__interval_matches(self.__wandb_val_metrics_record_interval):
                                self.log(f'val/{m}/{r}', mean_value, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                                data = {
                                    'epoch': self.current_epoch,
                                    'step': self.global_step,
                                    'mode': 'validate',
                                    'module': '',
                                    'module-type': '',
                                    'shape': '',
                                    'metric': f'{m}-{r}',
                                    'value': mean_value,
                                }
                                self.__training_metrics = append_row(self.__training_metrics, data)
                        
        # Log prediction images.
        if self.logger is not None:
            if self.current_epoch % self.__val_image_interval == 0 and (self.__val_max_image_batches is None or batch_idx < self.__val_max_image_batches):
                x = x.cpu().numpy()
                if y_hat.dtype != np.bool_:
                    # Convert from softmax to binary mask.
                    y_hat = torch.nn.functional.one_hot(y_hat.argmax(axis=1), num_classes=self.__n_channels)
                    y_hat = y_hat.moveaxis(-1, 1)
                    y_hat = y_hat.cpu().numpy().astype(bool)
                    y = y.cpu().numpy()

                class_labels = {
                    1: 'foreground'
                }
                for i, d in enumerate(descs):
                    for j, r in enumerate(self.__regions):
                        c = j + 1
                        # Skip channel if GT not present.
                        if not mask[i, c]:
                            continue

                        # Get images.
                        y_hat_vol, x_vol, y_vol = y_hat[i, c], x[i, 0], y[i, c]

                        # Get centre of extent of ground truth.
                        centre = centre_of_extent(y_vol)
                        if centre is None:
                            # Presumably data augmentation has pushed the label out of view.
                            continue

                        # Plot each orientation.
                        for a, c in enumerate(centre):
                            # Get 2D slice.
                            idxs = tuple([c if k == a else slice(0, x_vol.shape[i]) for k in range(3)])
                            y_hat_img, x_img, y_img = y_hat_vol[idxs], x_vol[idxs], y_vol[idxs]

                            # Fix orientation.
                            if a in (0, 1):     # Sagittal/coronal views.
                                y_hat_img = np.rot90(y_hat_img)
                                x_img = np.rot90(x_img)
                                y_img = np.rot90(y_img)
                            else:               # Axial view.
                                x_img = np.transpose(x_img)
                                y_img = np.transpose(y_img) 
                                y_hat_img = np.transpose(y_hat_img)

                            # Send image.
                            title = f'desc:{d}:region:{r}:axis:{a}'
                            caption = d,
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

        # Final loss reduction.
        # Should use unweighted loss for validation.
        loss = task_losses.mean()
        if self.__interval_matches(self.__wandb_val_loss_record_interval):
            self.log('val/loss', loss, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
        if self.__save_training_metrics and self.__interval_matches(self.__val_loss_record_interval):
            # Note that this will create lots of rows in 'training_metrics' that are identical
            # except for the 'values' column. This is because 'step' column does not increment
            # with validation batch number, unlike during training.
            # This is fine, our plotting code aggregates values anyway.
            self.save_loss(loss)

    def backward(self, task_losses):
        # Final loss reduction.
        loss = task_losses.mean()

        if self.__interval_matches(self.__wandb_loss_record_interval):
            self.log('train/loss', loss, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        if self.__save_training_metrics and self.__interval_matches(self.__loss_record_interval):
            self.save_loss(loss)

        # Calculate norms for Gradnorm. These norms should not be clipped as we want weight
        # updates to reflect actual norms.
        self.__hooks_enabled = False
        task_norms = []
        for l in task_losses:
            # Backprop directly from loss -> final shared params for Gradnorm loss.
            # Can't do clipping calc here, as we need full grads for Gradnorm weight updates.
            gs = torch.autograd.grad(l, self.__gn_balance_point, create_graph=True, retain_graph=True)
            # Calculate total norm - grads from both weights and bias tensors.
            task_norm = torch.stack([g.norm(2) for g in gs]).norm(2)
            task_norms.append(task_norm)
        task_norms = torch.stack(task_norms)

        self.__hooks_enabled = True

        mean_norm = task_norms.mean()
        # 'detach()' is used to stop the gradient from the target term from affecting task weights.
        # This could simply push task weights to zero to minimise 'L_grad'.
        L_grad_targets = (mean_norm * self.__rs ** self.__gn_alpha).detach()
        L_grad_tasks = self.__gn_loss_fn(task_norms - L_grad_targets)

        if self.logger is not None:
            self.log(f'train/gn-norm-mean', mean_norm, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
            data = {
                'epoch': self.current_epoch,
                'step': self.global_step,
                'mode': 'train',
                'module': '',
                'module-type': '',
                'shape': '',
                'metric': f'gradnorm-norm-mean',
                'value': mean_norm.item(),
            }
            self.__training_metrics = append_row(self.__training_metrics, data)

        for r, g, l, t in zip(self.__regions, task_norms, L_grad_tasks, L_grad_targets):
            if self.logger is not None:
                self.log(f'train/gn-norm/{r}', g, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
                self.log(f'train/gn-loss/{r}', l, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
                self.log(f'train/gn-loss-target/{r}', t, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

            if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'gradnorm-norm-{r}',
                    'value': g.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'gradnorm-loss-{r}',
                    'value': l.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': 'train',
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': f'gradnorm-loss-target-{r}',
                    'value': t.item(),
                }
                self.__training_metrics = append_row(self.__training_metrics, data)

        # Calculate scalar GradNorm loss.
        # Save 'L_grad' for after main backward pass, we don't want to update the task
        # weights until these values have been used for main backprop.
        self.__L_grad = L_grad_tasks.sum()

        if self.logger is not None:
            self.log(f'train/gn-loss', self.__L_grad, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
            data = {
                'epoch': self.current_epoch,
                'step': self.global_step,
                'mode': 'train',
                'module': '',
                'module-type': '',
                'shape': '',
                'metric': f'gradnorm-loss',
                'value': self.__L_grad.item(),
            }
            self.__training_metrics = append_row(self.__training_metrics, data)

        if self.__gn_clip_mult is not None:
            # Manually accumulate final param layer, 'autograd.grad' doesn't accumulate gradients. We can't
            # use 'autograd.backward' as there's no way to split by task and clip.
            accum_params = list(self.__network.get_submodule(self.__gn_clipping_accum_layer).parameters())
            task_grads = []
            for l in task_losses:
                grads = torch.autograd.grad(l, accum_params, create_graph=False, retain_graph=True)
                task_grads.append(grads)
            weight_grad, bias_grad = task_grads[0]
            for w, b in task_grads[1:]:
                weight_grad += w
                bias_grad += b
            accum_params[0].grad = weight_grad
            accum_params[1].grad = bias_grad

            # Get grads at clipping point.
            task_grads = []
            task_norms = []
            for l in task_losses:
                grad = torch.autograd.grad(l, [self.__gn_clipping_output], create_graph=False, retain_graph=True)[0]
                task_grads.append(grad)
                task_norms.append(grad.norm(2))
            task_norms = torch.stack(task_norms)

            for r, n in zip(self.__regions, task_norms):
                if self.logger is not None:
                    self.log(f'train/gn-clip-norm/{r}', n, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': 'train',
                        'module': '',
                        'module-type': '',
                        'shape': '',
                        'metric': f'gradnorm-clip-norm-{r}',
                        'value': n.item(),
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

            # Clip gradients.
            # mean_norm = task_norms.mean()
            # Calculate the mean if each task was excluded - this stops a large task from dragging up the mean.
            mean_norms = (task_norms.sum() - task_norms) / len(task_norms)

            if self.logger is not None:
                for r, n in zip(self.__regions, mean_norms):
                    self.log(f'train/gn-clip-norm-mean/{r}', n, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                    if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                        data = {
                            'epoch': self.current_epoch,
                            'step': self.global_step,
                            'mode': 'train',
                            'module': '',
                            'module-type': '',
                            'shape': '',
                            'metric': f'gradnorm-clip-norm-mean-{r}',
                            'value': n.item(),
                        }
                        self.__training_metrics = append_row(self.__training_metrics, data)

            # Perform clipping.
            task_thresholds = mean_norms * self.__gn_clip_mult
            should_clip = task_norms > task_thresholds
            clipped_norms = []
            for i in range(len(task_grads)):
                r = self.__regions[i]

                if self.logger is not None:
                    self.log(f'train/gn-clip/{r}', float(should_clip[i]), on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': 'train',
                        'module': '',
                        'module-type': '',
                        'shape': '',
                        'metric': f'gradnorm-clip-{r}',
                        'value': float(should_clip[i].item()),
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

                if not should_clip[i]:
                    clipped_norms.append(task_norms[i])
                    continue

                task_grads[i] = task_grads[i] * (task_thresholds[i] / task_norms[i])     # Scale to the threshold.
                clipped_norms.append(task_grads[i].norm(2))

            for r, n in zip(self.__regions, clipped_norms):
                if self.logger is not None:
                    self.log(f'train/gn-clip-norm-clipped/{r}', n, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                if self.__save_training_metrics and self.__interval_matches(self.__metrics_record_interval):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': 'train',
                        'module': '',
                        'module-type': '',
                        'shape': '',
                        'metric': f'gradnorm-clip-norm-clipped-{r}',
                        'value': n.item(),
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

            # Sum the clipped grads and backpropagate.
            grad = torch.stack(task_grads).sum(dim=0)
            self.__gn_clipping_output.backward(gradient=grad, retain_graph=True)
        else:
            loss.backward(retain_graph=True)

    def on_after_backward(self) -> None:
        if not self.__hooks_enabled:
            return

        if self.__gn_enabled:
            # Gradnorm weight updates.

            # This will backpropagate L_grad gradients wrt. task weights back to task weight tensors.
            self.__task_optimiser.zero_grad()   # Clear previously calculated gradients.
            # Channel-specific backward hooks were running, as 'L_grad' depends on the output gradients
            # of these final layers.
            self.__hooks_enabled = False
            # 'L_grad' depends on gradients of task-specific losses wrt. balance points, which in turn, depend
            # on task-specific losses, which in turn depend on all network parameters! If we blindly run 'L_grad.backward()'
            # here, we'll pass through the entire network, which we don't want until 'loss.backward()' is called.
            # Also we can't detach our GradNorm loss graph from the main network, as we need to backprop to task
            # weights.
            grad_w = torch.autograd.grad(self.__L_grad, self.__gn_task_weights, retain_graph=False)[0]
            self.__hooks_enabled = True
            self.__gn_task_weights.grad = grad_w
            self.__task_optimiser.step() # Update task weights.

            # Renormalise task weights so that no effective learning rate shift can occur.
            self.__gn_task_weights.data = len(self.__gn_task_weights) * self.__gn_task_weights.data / self.__gn_task_weights.data.sum()

        if self.__save_training_metrics and self.__interval_matches(self.__metrics_save_interval):
            # Save training metrics.
            filepath = os.path.join(config.directories.models, self.__name[0], self.__name[1], 'training-metrics.csv')
            save_csv(self.__training_metrics, filepath)

    def save_loss(
        self,
        loss: torch.Tensor) -> None:
        mode = 'train' if self.training else 'validate'
        data = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'mode': mode,
            'module': '',
            'module-type': '',
            'shape': '',
            'metric': 'loss',
            'value': loss.item(),
        }
        self.__training_metrics = append_row(self.__training_metrics, data)

    def __interval_matches(
        self,
        interval: TrainingInterval) -> bool:
        # Parse interval.
        terms = interval.split(':')
        n_terms = len(terms)
        assert n_terms in (2, 4)

        if len(terms) == 2:
            if terms[0] == 'step':
                return self.__step_matches(int(terms[1]))
            elif terms[0] == 'epoch':
                epoch = int(terms[1]) if terms[1].isdigit() else terms[1]
                return self.__epoch_matches(epoch)
        elif len(terms) == 4:
            epoch = int(terms[1]) if terms[1].isdigit() else terms[1]
            return self.__epoch_matches(epoch) and self.__step_matches(int(terms[3]))

    def __epoch_matches(
        self,
        epoch: Union[int, Literal['start', 'end']]) -> bool:
        if isinstance(epoch, int):
            return self.current_epoch % epoch == 0
        elif epoch == 'start':
            if self.trainer.train_dataloader is None:     # Not set during "sanity-checking" validation steps.
                return False
            return self.global_step % len(self.trainer.train_dataloader) == 0
        elif epoch == 'end':
            if self.trainer.train_dataloader is None:     # Not set during "sanity-checking" validation steps.
                return False
            return self.global_step % len(self.trainer.train_dataloader) == len(self.trainer.train_dataloader) - 1

    def __step_matches(
        self,
        step: int) -> bool:
        return self.global_step % step == 0

    def __register_clipping_hook(self) -> None:
        def hook(module, _, output) -> None:
            self.__gn_clipping_output = output
        m = self.__network.get_submodule(self.__gn_clipping_layer)
        m.register_forward_hook(hook)
    
    def __register_training_metrics_hooks(self) -> None:
        # Create training stats dataframe.
        cols = {
            'epoch': int,
            'step': int,
            'mode': str,
            'module': str,
            'module-type': str,
            'shape': str,
            'metric': str,
            'value': float,
        }
        self.__training_metrics = pd.DataFrame(columns=cols.keys())

        for n, m in self.__network.named_modules():
            if n in self.__leaf_modules:
                # Register forward/backward module hooks to get parameters, outputs
                # and gradients w.r.t outputs.
                m.register_forward_hook(self.__get_forward_hook(n))
                m.register_full_backward_hook(self.__get_backward_hook(n))

            if n in self.__param_modules:
                params = list(m.parameters())
                assert len(params) == 2
                param_names = ['weight', 'bias']
                for pn, p in zip(param_names, params):
                    p.register_hook(self.__get_parameter_hook(n, m.__class__.__name__, pn))

    def __get_parameter_hook(
        self,
        module_name: str,
        module_type: str,
        param_name: str) -> Callable:
        def hook(grad) -> None:
            if self.__interval_matches(self.__metrics_record_interval):
                mode = 'train' if self.training else 'validate'

                # Save parameter gradient stats.
                metrics = [
                    f'gradient-{param_name}-min',
                    f'gradient-{param_name}-max',
                    f'gradient-{param_name}-mean',
                    f'gradient-{param_name}-std',
                    f'gradient-{param_name}-l2'
                ]
                values = [
                    grad.min().item(),
                    grad.max().item(),
                    grad.mean().item(),
                    grad.std().item(),
                    grad.norm(2).item(),
                ]
                
                for m, v in zip(metrics, values):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': mode,
                        'module': module_name,
                        'module-type': module_type,
                        'shape': str(list(grad.shape)),
                        'metric': m,
                        'value': v,
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

        return hook

    def __get_forward_hook(
        self,
        name: str) -> Callable:
        def hook(module, _, output) -> None:
            if self.__interval_matches(self.__metrics_record_interval):
                mode = 'train' if self.training else 'validate'

                # Add output metrics.
                metrics = [
                    'output-min',
                    'output-max',
                    'output-mean',
                    'output-std',
                    'output-l2',
                ]
                values = [
                    output.min().item(),
                    output.max().item(),
                    output.mean().item(),
                    output.std().item(),
                    output.norm(2).item(),
                ]

                # Add parameter metrics.
                if name in self.__param_modules:
                    params = list(module.parameters())
                    assert len(params) == 2
                    param_names = ['weight', 'bias']
                    for n, p in zip(param_names, params):
                        metrics += [
                            f'{n}-min',
                            f'{n}-max',
                            f'{n}-mean',
                            f'{n}-std',
                            f'{n}-l2',
                        ]
                        values += [
                            p.min().item(),
                            p.max().item(),
                            p.mean().item(),
                            p.std().item(),
                            p.norm(2).item(),
                        ]

                for m, v, in zip(metrics, values):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': mode,
                        'module': name,
                        'module-type': module.__class__.__name__,
                        'shape': str(list(output.shape)),
                        'metric': m,
                        'value': v,
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

        return hook

    def __get_backward_hook(
        self,
        name: str) -> Callable:
        # 'grad_input' is not straightforward - https://discuss.pytorch.org/t/exact-meaning-of-grad-input-and-grad-output/14186/7.
        def hook(module, _, grad_output) -> None:
            if not self.__hooks_enabled:
                return
            
            # Clip individual task norms.

            if self.__interval_matches(self.__metrics_record_interval):
                mode = 'train' if self.training else 'validate'

                # 'grad_output' is a tuple because some layers (e.g. torch.split) return multiple outputs.
                if len(grad_output) != 1:
                    raise ValueError(f"Module '{module.__class__.__name__} has 'grad_output' of length {len(grad_output)}.")
                grad_output = grad_output[0]

                # Save gradient stats.
                metrics = [
                    'gradient-output-min',
                    'gradient-output-max',
                    'gradient-output-mean',
                    'gradient-output-std',
                    'gradient-output-l2',
                ]
                values = [
                    grad_output.min().item(),
                    grad_output.max().item(),
                    grad_output.mean().item(),
                    grad_output.std().item(),
                    grad_output.norm(2).item(),
                ]
                for m, v, in zip(metrics, values):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'mode': mode,
                        'module': name,
                        'module-type': module.__class__.__name__,
                        'shape': str(list(grad_output.shape)),
                        'metric': m,
                        'value': v,
                    }
                    self.__training_metrics = append_row(self.__training_metrics, data)

        return hook
