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

from ..architectures import create_mednext_v1, layer_summary, UNet3D
from .lightning_modules import replace_ckpt_alias

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        regions: PatientRegions,
        arch: str = 'unet3d:m',
        loss_fn: str = 'dice',
        loss_smoothing: float = 0.1,
        lr_init: float = 1e-3,
        name: Optional[ModelName] = None,
        save_training_metrics: bool = False,
        loss_record_interval: TrainingInterval = 'step:1',     # Loss should typically be recorded more frequently - it's not very expensive.
        metrics_record_interval: TrainingInterval = 'step:5',
        metrics_save_interval: TrainingInterval = 'epoch:end',
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
            self.__loss_fn = DiceLoss(smoothing=loss_smoothing)
        elif loss_fn == 'dice-label-smoothing':
            self.__loss_fn = DiceWithLabelSmoothingLoss(smoothing=loss_smoothing)
        elif loss_fn == 'dml1':
            self.__loss_fn = DML1Loss(smoothing=loss_smoothing)
        elif loss_fn == 'dml2':
            self.__loss_fn = DML2Loss(smoothing=loss_smoothing)
        elif loss_fn == 'tversky':
            self.__loss_fn = TverskyLoss()
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
        self.__wandb_metrics_record_interval = wandb_metrics_record_interval
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
                raise ValueError(f"Can't load Segmenter '{model}', has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model = replace_ckpt_alias(model)
        filepath = os.path.join(config.directories.models, model[0], model[1], f"{model[2]}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"Segmenter '{model}' not found.")

        # Load checkpoint.
        segmenter = Segmenter.load_from_checkpoint(filepath, name=model, **kwargs)

        return segmenter

    def configure_optimizers(self):
        # self.__optimiser = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.__weight_decay) 
        self.__optimiser = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        logging.info(f"Using optimiser '{self.__optimiser}' with learning rate '{self.lr}'.")

        opt = {
            'optimizer': self.__optimiser,
            'monitor': 'val/loss'
        }
        opt['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimiser, factor=0.8, mode='min', patience=100, verbose=True)
        logging.info(f"Using optimiser scheduler 'ReduceLROnPlateau' with factor '0.8' and patient '100'.")

        return opt

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        return self.__network(x)

    def training_step(self, batch, batch_idx):
        desc, x, y, mask = batch
        n_batch_items = len(desc)

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
                self.log(f'train/loss/region/{r}', l, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Final loss reduction.
        loss = loss.mean()
        if self.__interval_matches(self.__wandb_loss_record_interval):
            self.log('train/loss', loss, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)
        if self.__save_training_metrics and self.__interval_matches(self.__loss_record_interval):
            self.save_loss(loss)

        # Log learning rate as the default logger doesn't log against the epoch.
        self.log('train/lr', self.lr, on_epoch=self.__wandb_log_on_epoch, on_step=not self.__wandb_log_on_epoch)

        # Report metrics.
        if (self.logger is not None and self.__interval_matches(self.__wandb_metrics_record_interval)) or \
            self.__interval_matches(self.__metrics_record_interval):
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
                            tp = true_positives(y_hat_c, y_c)
                            fp = false_positives(y_hat_c, y_c)
                            fn = false_negatives(y_hat_c, y_c)
                            tn = true_negatives(y_hat_c, y_c)
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

                            if self.__interval_matches(self.__metrics_record_interval):
                                data = {
                                    'epoch': self.current_epoch,
                                    'step': self.global_step,
                                    'module': '',
                                    'module-type': '',
                                    'shape': '',
                                    'metric': f'train-{m}-{r}',
                                    'value': mean_value,
                                }
                                self.__training_metrics = append_row(self.__training_metrics, data)

        return loss

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
                            tp = true_positives(y_hat_c, y_c)
                            fp = false_positives(y_hat_c, y_c)
                            fn = false_negatives(y_hat_c, y_c)
                            tn = true_negatives(y_hat_c, y_c)
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

                            if self.__interval_matches(self.__metrics_record_interval):
                                data = {
                                    'epoch': self.current_epoch,
                                    'step': self.global_step,
                                    'module': '',
                                    'module-type': '',
                                    'shape': '',
                                    'metric': f'validate-{m}-{r}',
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

    def on_after_backward(self) -> None:
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
            if n == '_UNet3D__layers.62._LayerWrapper__layer':
                m.register_forward_hook(self.__get_final_conv_diff_hook(n))

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
                ]
                values = [
                    grad.min().item(),
                    grad.max().item(),
                    grad.mean().item(),
                    grad.std().item(),
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
                    f'output-min',
                    f'output-max',
                    f'output-mean',
                    f'output-std',
                ]
                values = [
                    output.min().item(),
                    output.max().item(),
                    output.mean().item(),
                    output.std().item(),
                ]

                # Add parameter metrics.
                if module.__class__.__name__ in self.__param_modules:
                    params = list(module.parameters())
                    assert len(params) == 2
                    param_names = ['weight', 'bias']
                    for n, p in zip(param_names, params):
                        metrics += [
                            f'parameter-{n}-min',
                            f'parameter-{n}-max',
                            f'parameter-{n}-mean',
                            f'parameter-{n}-std',
                        ]
                        values += [
                            p.min().item(),
                            p.max().item(),
                            p.mean().item(),
                            p.std().item(),
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
            if self.__interval_matches(self.__metrics_record_interval):
                mode = 'train' if self.training else 'validate'

                # Add abs diff between foreground/background channels.
                metrics = [
                    f'output-diff-min',
                    f'output-diff-max',
                    f'output-diff-mean',
                    f'output-diff-std',
                ]
                diff = torch.abs(output[:, 1] - output[:, 0])
                values = [
                    diff.min().item(),
                    diff.max().item(),
                    diff.mean().item(),
                    diff.std().item(),
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
                ]
                values = [
                    grad_output.min().item(),
                    grad_output.max().item(),
                    grad_output.mean().item(),
                    grad_output.std().item(),
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
