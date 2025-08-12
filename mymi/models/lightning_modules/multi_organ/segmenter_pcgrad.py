import numpy as np
import os
import pytorch_lightning as pl
import sys
import torch
from typing import *

pcgrad_path = os.path.join(os.environ['CODE'], 'Pytorch-PCGrad')
sys.path.append(pcgrad_path)
from pcgrad import PCGrad

from mymi import config
from mymi.geometry import fov_centre
from mymi import logging
from mymi.losses import *
from mymi.metrics import *
from mymi.typing import *
from mymi.utils import *

from ...architectures import create_mednext_v1, layer_summary, UNet3D
from ...models import replace_ckpt_alias

class SegmenterPCGrad(pl.LightningModule):
    def __init__(
        self,
        regions: Regions,
        arch: str = 'unet3d:m',
        loss_fn: str = 'dice',
        loss_smoothing: float = 0.1,
        lr_init: float = 1e-3,
        metrics_save_interval: TrainingInterval = 'epoch:end',
        name: Optional[ModelName] = None,
        loss_record_interval: TrainingInterval = 'step:1',     # Loss should typically be recorded more frequently - it's not very expensive.
        pc_reduction: Literal['mean', 'sum'] = 'mean',
        save_training_metrics: bool = False,
        train_metrics_record_interval: TrainingInterval = 'step:5',
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5,
        val_image_interval: int = 50,
        val_loss_record_interval: TrainingInterval = 'epoch:1',   # Requires its own parameter as step could have any value at the end of the epoch.
        val_max_image_batches: Optional[int] = 3,
        val_metrics_record_interval: TrainingInterval = 'epoch:1',   # Requires own parameter.
        wandb_log_on_epoch: bool = True,    # If True, regardless of our logging frequency, wandb will accumulate values over the epoch before logging.
        wandb_loss_record_interval: TrainingInterval = 'step:1',
        wandb_train_metrics_record_interval: TrainingInterval = 'step:5',
        wandb_val_loss_record_interval: TrainingInterval = 'epoch:1',
        wandb_val_metrics_record_interval: TrainingInterval = 'epoch:1',    # Requires own parameter.
        **kwargs) -> None:
        super().__init__()
        self.__loss_record_interval = loss_record_interval
        self.__pc_reduction = pc_reduction
        self.__train_metrics_record_interval = train_metrics_record_interval
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
        if arch == 'unet3d:m':
            logging.info(f"Using UNet3D (M) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, **kwargs)
            self.__leaf_modules = layer_summary(arch, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
        elif arch == 'unet3d:l':
            logging.info(f"Using UNet3D (L) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, n_features=64, **kwargs)
            self.__leaf_modules = layer_summary(arch, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
        elif arch == 'unet3d:xl':
            logging.info(f"Using UNet3D (XL) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, n_features=128, **kwargs)
            self.__leaf_modules = layer_summary(arch, self.__n_channels, params_only=False, leafs_only=True)['module'].tolist()
            self.__param_modules = layer_summary(arch, self.__n_channels, params_only=True, leafs_only=True)['module'].tolist()
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
        self.__wandb_train_metrics_record_interval = wandb_train_metrics_record_interval
        self.__val_image_interval = val_image_interval
        self.__val_max_image_batches = val_max_image_batches
        self.__wandb_val_metrics_record_interval = wandb_val_metrics_record_interval
        self.__wandb_val_loss_record_interval = wandb_val_loss_record_interval

        if self.__save_training_metrics:
            logging.info("Saving training metrics.")
            self.__register_training_metrics_hooks()

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
                raise ValueError(f"Can't load SegmenterPCGrad '{model}', has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model = replace_ckpt_alias(model)
        filepath = os.path.join(config.directories.models, model[0], model[1], f"{model[2]}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"SegmenterPCGrad '{model}' not found.")

        # Load checkpoint.
        segmenter = SegmenterPCGrad.load_from_checkpoint(filepath, name=model, **kwargs)

        return segmenter

    def configure_optimizers(self):
        # self.__optimiser = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.lr)
        logging.info(f"Using optimiser '{optimiser}' with learning rate '{self.lr}'.")
        self.__optimiser = PCGrad(optimiser, reduction=self.__pc_reduction)
        logging.info(f"Using PCGrad with reduction '{self.__pc_reduction}'.")
        # opt = {
        #     'optimizer': self.__optimiser,
        #     'monitor': 'val/loss'
        # }
        # opt['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimiser, factor=0.8, mode='min', patience=100, verbose=True)
        # logging.info(f"Using optimiser scheduler 'ReduceLROnPlateau' with factor '0.8' and patient '100'.")

        # We'll handle the step ourselves.
        return None

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
        def log_fn(metric: str, value: float) -> None:
            mode = 'train' if self.training else 'validate'
            self.log(f'{mode}/{metric}', value, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
            if self.__save_training_metrics and self.__interval_matches(self.__loss_record_interval):
                data = {
                    'epoch': self.current_epoch,
                    'step': self.global_step,
                    'mode': mode,
                    'module': '',
                    'module-type': '',
                    'shape': '',
                    'metric': metric,
                    'value': value,
                }
                self.__training_metrics = append_row(self.__training_metrics, data)
        kwargs = dict(
            log_fn=log_fn,
            include_background=False,
            mask=mask,
            reduce_channels=False,
        )
        task_losses = self.__loss_fn(y_hat, y, **kwargs) 

        # Log region losses.
        if self.logger is not None:
            for r, l in zip(self.__regions, task_losses):
                self.log(f'train/loss/region/{r}', l, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Log learning rate as the default logger doesn't log against the epoch.
        self.log('train/lr', self.lr, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Report metrics.
        if (self.logger is not None and self.__interval_matches(self.__wandb_train_metrics_record_interval)) or \
            self.__interval_matches(self.__train_metrics_record_interval):
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
                mean_dices = []
                for i, r in enumerate(self.__regions):
                    c = i + 1
                    dices = []
                    tprs = []
                    fprs = []
                    fnrs = []
                    tnrs = []
                    for b in range(n_batch_items):
                        if mask[b, c]:
                            # Get batch item.
                            y_hat_c = y_hat[b, c]
                            y_c = y[b, c]   

                            # Calculate dice.
                            d = dice(y_hat_c, y_c)
                            dices.append(d)

                            # Calculate classification stats.
                            tpr = true_positive_rate(y_hat_c, y_c)
                            fpr = false_positive_rate(y_hat_c, y_c)
                            fnr = false_negative_rate(y_hat_c, y_c)
                            tnr = true_negative_rate(y_hat_c, y_c)
                            tprs.append(tpr)
                            fprs.append(fpr)
                            fnrs.append(fnr)
                            tnrs.append(tnr)
                    if len(dices) > 0:  
                        mean_dices.append(np.mean(dices))

                    metrics = ['dice', 'tpr', 'fpr', 'fnr', 'tnr']
                    values = [dices, tprs, fprs, fnrs, tnrs]
                    for m, v in zip(metrics, values):
                        if len(v) > 0:
                            mean_value = np.mean(v)
                            if self.logger is not None and self.__interval_matches(self.__wandb_train_metrics_record_interval):
                                self.log(f'train/{m}/{r}', mean_value, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                            if self.__save_training_metrics and self.__interval_matches(self.__train_metrics_record_interval):
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

                # Log mean dice across all regions.
                if len(mean_dices) > 0:
                    mean_dice = np.mean(mean_dices)
                    if self.logger is not None and self.__interval_matches(self.__wandb_train_metrics_record_interval):
                        self.log(f'train/dice', mean_dice, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                    if self.__save_training_metrics and self.__interval_matches(self.__train_metrics_record_interval):
                        data = {
                            'epoch': self.current_epoch,
                            'step': self.global_step,
                            'mode': 'train',
                            'module': '',
                            'module-type': '',
                            'shape': '',
                            'metric': 'dice',
                            'value': mean_dice,
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
        )
        loss = self.__loss_fn(y_hat, y, **kwargs) 

        # Log region losses.
        if self.logger is not None:
            for r, l in zip(self.__regions, loss):
                self.log(f'val/loss/region/{r}', l, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Final loss reduction.
        loss = loss.mean()
        if self.__interval_matches(self.__wandb_val_loss_record_interval):
            self.log('val/loss', loss, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
        if self.__save_training_metrics and self.__interval_matches(self.__val_loss_record_interval):
            # Note that this will create lots of rows in 'training_metrics' that are identical
            # except for the 'values' column. This is because 'step' column does not increment
            # with validation batch number, unlike during training.
            # This is fine, our plotting code aggregates values anyway.
            self.save_loss(loss)

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
                mean_dices = []
                for i, r in enumerate(self.__regions):
                    c = i + 1
                    dices = []
                    tprs = []
                    fprs = []
                    fnrs = []
                    tnrs = []
                    for b in range(n_batch_items):
                        if mask[b, c]:
                            # Get batch item.
                            y_hat_c = y_hat[b, c]
                            y_c = y[b, c]   

                            # Calculate dice.
                            d = dice(y_hat_c, y_c)
                            dices.append(d)

                            # Calculate classification stats.
                            tpr = true_positive_rate(y_hat_c, y_c)
                            fpr = false_positive_rate(y_hat_c, y_c)
                            fnr = false_negative_rate(y_hat_c, y_c)
                            tnr = true_negative_rate(y_hat_c, y_c)
                            tprs.append(tpr)
                            fprs.append(fpr)
                            fnrs.append(fnr)
                            tnrs.append(tnr)
                    if len(dices) > 0:
                        mean_dices.append(np.mean(dices))

                    metrics = ['dice', 'tpr', 'fpr', 'fnr', 'tnr']
                    values = [dices, tprs, fprs, fnrs, tnrs]
                    for m, v in zip(metrics, values):
                        if len(v) > 0:
                            mean_value = np.mean(v)
                            if self.logger is not None and self.__interval_matches(self.__wandb_val_metrics_record_interval):
                                self.log(f'val/{m}/{r}', mean_value, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                            if self.__save_training_metrics and self.__interval_matches(self.__val_metrics_record_interval):
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

                # Log mean dice across all regions.
                if len(mean_dices) > 0:
                    mean_dice = np.mean(mean_dices)
                    if self.logger is not None and self.__interval_matches(self.__wandb_val_metrics_record_interval):
                        self.log(f'val/dice', mean_dice, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

                    if self.__save_training_metrics and self.__interval_matches(self.__val_metrics_record_interval):
                        data = {
                            'epoch': self.current_epoch,
                            'step': self.global_step,
                            'mode': 'validate',
                            'module': '',
                            'module-type': '',
                            'shape': '',
                            'metric': 'dice',
                            'value': mean_dice,
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
                        centre = fov_centre(y_vol)
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

    def optimizer_zero_grad(self, *_) -> None:
        # PCGrad calls 'zero_grad'.
        pass

    def backward(
        self,
        task_losses) -> None:
        self.__optimiser.pc_backward(task_losses)

        # Final loss reduction.
        loss = task_losses.mean()
        if self.__interval_matches(self.__wandb_loss_record_interval):
            self.log('train/loss', loss, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
        if self.__save_training_metrics and self.__interval_matches(self.__loss_record_interval):
            self.save_loss(loss)

    def optimizer_step(self, epoch, batch_idx, optimizer, closure=None) -> None:
        # Evaluate the training step and backward 'optimizer_closure' manually as PCGrad optimiser won't do it.
        # This is because lightning delegates to the optimiser to support complex optimiser LBFGS that require
        # the function to be evaluated multiple times.

        if closure is not None:
            closure()
        else:
            print('optimiser closure is None')
        self.__optimiser.step()

    def on_train_batch_end(self, *_) -> None:
        if self.__save_training_metrics and self.__interval_matches(self.__metrics_save_interval):
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

            # Custom hooks.
            channel_grad_modules = [
                '_UNet3D__layers.62._LayerWrapper__layer',
                '_UNet3D__layers.63._LayerWrapper__layer',
            ]
            if n in channel_grad_modules:
                m.register_full_backward_hook(self.__get_channel_grad_backward_hook(n))

    def __get_parameter_hook(
        self,
        module_name: str,
        module_type: str,
        param_name: str) -> Callable:
        def hook(grad) -> None:
            mode = 'train' if self.training else 'validate'
            if (mode == 'train' and self.__interval_matches(self.__train_metrics_record_interval)) or \
                (mode == 'validate' and self.__interval_matches(self.__val_metrics_record_interval)):
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
            mode = 'train' if self.training else 'validate'
            if (mode == 'train' and self.__interval_matches(self.__train_metrics_record_interval)) or \
                (mode == 'validate' and self.__interval_matches(self.__val_metrics_record_interval)):

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

    def __get_final_conv_diff_hook(
        self,
        name: str) -> Callable:
        def hook(module, _, output) -> None:
            mode = 'train' if self.training else 'validate'
            if (mode == 'train' and self.__interval_matches(self.__train_metrics_record_interval)) or \
                (mode == 'validate' and self.__interval_matches(self.__val_metrics_record_interval)):

                # Add abs diff between foreground/background channels.
                metrics = [
                    'output-diff-min',
                    'output-diff-max',
                    'output-diff-mean',
                    'output-diff-std',
                    'output-diff-l2',
                ]
                diff = torch.abs(output[:, 1] - output[:, 0])
                values = [
                    diff.min().item(),
                    diff.max().item(),
                    diff.mean().item(),
                    diff.std().item(),
                    diff.norm(2).item(),
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
            mode = 'train' if self.training else 'validate'
            if (mode == 'train' and self.__interval_matches(self.__train_metrics_record_interval)) or \
                (mode == 'validate' and self.__interval_matches(self.__val_metrics_record_interval)):

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

    def __get_channel_grad_backward_hook(
        self,
        name: str) -> Callable:
        def hook(module, _, grad_output) -> None:
            mode = 'train' if self.training else 'validate'
            if (mode == 'train' and self.__interval_matches(self.__train_metrics_record_interval)) or \
                (mode == 'validate' and self.__interval_matches(self.__val_metrics_record_interval)):

                # 'grad_output' is a tuple because some layers (e.g. torch.split) return multiple outputs.
                if len(grad_output) != 1:
                    raise ValueError(f"Module '{module.__class__.__name__} has 'grad_output' of length {len(grad_output)}.")
                grad_output = grad_output[0]

                # Save gradient stats.
                # What happens when a sample is missing a particular channel, e.g. optic chiasm.
                # We've masked out that channel's contribution to the loss function, so gradients will
                # be zero for that step. They'll still flow backwards though, and it could be interesting
                # as this will show us how much class imbalance affects the gradients.
                n_channels = grad_output.shape[1]
                assert n_channels == self.__n_channels
                channel_names = ['background'] + self.__regions
                for c, n in zip(range(n_channels), channel_names):
                    # Save total gradient stats.
                    grad_c = grad_output[:, c]
                    metrics = [
                        f'gradient-output-{n}-min',
                        f'gradient-output-{n}-max',
                        f'gradient-output-{n}-mean',
                        f'gradient-output-{n}-std',
                        f'gradient-output-{n}-l2',
                    ]
                    values = [
                        grad_c.min().item(),
                        grad_c.max().item(),
                        grad_c.mean().item(),
                        grad_c.std().item(),
                        grad_c.norm(2).item(),
                    ]
                    for m, v, in zip(metrics, values):
                        data = {
                            'epoch': self.current_epoch,
                            'step': self.global_step,
                            'mode': mode,
                            'module': name,
                            'module-type': module.__class__.__name__,
                            'shape': str(list(grad_c.shape)),
                            'metric': m,
                            'value': v,
                        }
                        self.__training_metrics = append_row(self.__training_metrics, data)

                    # Save TP/TN/FP/FN gradient stats.
                    # Need to actually select the values.
                    TP = torch.masked_select(grad_c, self.__tp_mask[:, c])
                    TN = torch.masked_select(grad_c, self.__tn_mask[:, c])
                    FP = torch.masked_select(grad_c, self.__fp_mask[:, c])
                    FN = torch.masked_select(grad_c, self.__fn_mask[:, c])
                    groups = ['tp', 'tn', 'fp', 'fn']
                    dists = [TP, TN, FP, FN]
                    for g, d in zip(groups, dists):
                        metrics = [
                            f'gradient-output-{n}-{g}-min',
                            f'gradient-output-{n}-{g}-max',
                            f'gradient-output-{n}-{g}-mean',
                            f'gradient-output-{n}-{g}-std',
                            f'gradient-output-{n}-{g}-l2',
                        ]
                        if d.numel() == 0:
                            values = [np.nan, np.nan, np.nan, np.nan, np.nan]
                        else:
                            values = [
                                d.min().item(),
                                d.max().item(),
                                d.mean().item(),
                                d.std().item(),
                                d.norm(2).item(),
                            ]
                        for m, v, in zip(metrics, values):
                            data = {
                                'epoch': self.current_epoch,
                                'step': self.global_step,
                                'mode': mode,
                                'module': name,
                                'module-type': module.__class__.__name__,
                                'shape': str(list(grad_c.shape)),
                                'metric': m,
                                'value': v,
                            }
                            self.__training_metrics = append_row(self.__training_metrics, data)

        return hook
