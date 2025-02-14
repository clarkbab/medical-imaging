import numpy as np
import os
import pytorch_lightning as pl
import torch
from typing import *

from mymi import config
from mymi.geometry import centre_of_extent
from mymi import logging
from mymi.losses import DiceLoss, TverskyLoss
from mymi.metrics import dice
from mymi.models import layer_summary, replace_ckpt_alias
from mymi.models.architectures import create_mednext_v1, UNet3D
from mymi.typing import *
from mymi.utils import *

class Segmenter(pl.LightningModule):
    def __init__(
        self,
        regions: PatientRegions,
        arch: str = 'unet3d:M',
        layer_stats_record_interval: TrainingInterval = 'step:5',
        layer_stats_save_interval: TrainingInterval = 'epoch:end',
        log_on_epoch: bool = True,
        log_on_step: bool = False,
        lr_init: float = 1e-3,
        name: Optional[ModelName] = None,
        save_layer_stats: bool = True,
        train_metric_batch_interval: int = 5,
        train_metric_epoch_interval: int = 20,
        val_image_interval: int = 50,
        val_max_image_batches: Optional[int] = 3,
        val_metric_batch_interval: int = 1,
        val_metric_epoch_interval: int = 5,
        **kwargs) -> None:
        super().__init__()
        self.__layer_stats_record_interval = layer_stats_record_interval
        self.__layer_stats_save_interval = layer_stats_save_interval
        self.__log_on_epoch = log_on_epoch
        self.__log_on_step = log_on_step
        # self.__loss_fn = TverskyLoss()
        self.__loss_fn = DiceLoss()
        self.lr = lr_init        # 'self.lr' is default key that LR finder looks for on module.
        self.__name = name
        self.__regions = regions
        self.__n_channels = len(self.__regions) + 1
        if arch == 'unet3d:M':
            logging.info(f"Using UNet3D (M) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels)
        elif arch == 'unet3d:L':
            logging.info(f"Using UNet3D (L) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, n_features=64)
        elif arch == 'unet3d:XL':
            logging.info(f"Using UNet3D (XL) with {self.__n_channels} channels.")
            self.__network = UNet3D(self.__n_channels, n_features=128)
        elif arch == 'mednext:S':
            logging.info(f"Using MedNeXt (S) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'S')
        elif arch == 'mednext:B':
            logging.info(f"Using MedNeXt (B) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'B')
        elif arch == 'mednext:M':
            logging.info(f"Using MedNeXt (M) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'M')
        elif arch == 'mednext:L':
            logging.info(f"Using MedNeXt (L) with {self.__n_channels} channels.")
            self.__network = create_mednext_v1(1, self.__n_channels, 'L')
        self.__save_layer_stats = save_layer_stats
        self.__train_metric_epoch_interval = train_metric_epoch_interval
        self.__train_metric_batch_interval = train_metric_batch_interval
        self.__val_image_interval = val_image_interval
        self.__val_max_image_batches = val_max_image_batches
        self.__val_metric_epoch_interval = val_metric_epoch_interval
        self.__val_metric_batch_interval = val_metric_batch_interval

        if self.__save_layer_stats:
            self.__register_layer_stats_hooks()

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
                raise ValueError(f"Can't load MedNeXt '{model}', has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model = replace_ckpt_alias(model)
        filepath = os.path.join(config.directories.models, model[0], model[1], f"{model[2]}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"MedNeXt '{model}' not found.")

        # Load checkpoint.
        segmenter = Segmenter.load_from_checkpoint(filepath, name=model, **kwargs)

        return segmenter

    def configure_optimizers(self):
        # self.__optimiser = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.__weight_decay) 
        self.__optimiser = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        logging.info(f"Using optimiser '{self.__optimiser}'.")
        opt = {
            'optimizer': self.__optimiser,
            'monitor': 'val/loss'
        }
        opt['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.__optimiser, factor=0.8, mode='min', patience=100, verbose=True)

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
                self.log(f'train/loss/region/{r}', l, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Final loss reduction.
        loss = loss.mean()
        self.log('train/loss', loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Log learning rate as the default logger doesn't log against the epoch.
        self.log('train/lr', self.lr, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Report metrics.
        if self.logger is not None:
            if self.current_epoch % self.__train_metric_epoch_interval == 0 and batch_idx % self.__train_metric_batch_interval == 0:
                # Convert from softmax to binary mask.
                y_hat = torch.nn.functional.one_hot(y_hat.argmax(axis=1), num_classes=self.__n_channels)
                y_hat = y_hat.moveaxis(-1, 1)
                y_hat = y_hat.cpu().numpy().astype(bool)
                y = y.cpu().numpy()

                # Calculate mean (over batch) dice per region.
                for i, r in enumerate(self.__regions):
                    c = i + 1
                    dices = []
                    for b in range(n_batch_items):
                        if mask[b, c]:
                            y_hat_c = y_hat[b, c]
                            y_c = y[b, c]   
                            d = dice(y_hat_c, y_c)
                            dices.append(d)

                    if len(dices) > 0:
                        mean_dice = np.mean(dices)
                        self.log(f'train/dice/{r}', mean_dice, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

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
                self.log(f'val/loss/region/{r}', l, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Final loss reduction.
        loss = loss.mean()
        self.log('val/loss', loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Record gpu usage.
        for i, usage_mb in enumerate(gpu_usage_nvml()):
            self.log(f'gpu/{i}', usage_mb, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Report metrics.
        if self.logger is not None:
            if self.current_epoch % self.__val_metric_epoch_interval == 0 and batch_idx % self.__val_metric_batch_interval == 0:
                # Convert from softmax to binary mask.
                y_hat = torch.nn.functional.one_hot(y_hat.argmax(axis=1), num_classes=self.__n_channels)
                y_hat = y_hat.moveaxis(-1, 1)
                y_hat = y_hat.cpu().numpy().astype(bool)
                y = y.cpu().numpy()

                # Calculate mean (over batch) dice per region.
                region_mean_dices = []
                for i, r in enumerate(self.__regions):
                    c = i + 1
                    region_dices = []
                    for b in range(n_batch_items):
                        if mask[b, c]:
                            y_hat_c = y_hat[b, c]
                            y_c = y[b, c]   
                            d = dice(y_hat_c, y_c)
                            region_dices.append(d)

                    if len(region_dices) > 0:
                        region_mean_dice = np.mean(region_dices)
                        region_mean_dices.append(region_mean_dice)
                        self.log(f'val/dice/{r}', region_mean_dice, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
                
                if len(region_mean_dices) > 0:
                    mean_dice = np.mean(region_mean_dices)
                    self.log(f'val/dice', mean_dice, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
            elif self.current_epoch == 0:
                # Initialise 'val/dice' metric in case we're using this for checkpointing.
                self.log(f'val/dice', 0, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)
                        
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
        # Save layer stats.
        if self.__save_layer_stats and self.__interval_matches(self.__layer_stats_save_interval):
            filepath = os.path.join(config.directories.models, self.__name[0], self.__name[1], 'output-stats.csv')
            save_csv(self.__output_stats, filepath)
            filepath = os.path.join(config.directories.models, self.__name[0], self.__name[1], 'gradient-stats.csv')
            save_csv(self.__gradient_stats, filepath)

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
            return self.global_step % len(self.trainer.train_dataloader) == 0
        elif epoch == 'end':
            return self.global_step % len(self.trainer.train_dataloader) == len(self.trainer.train_dataloader) - 1

    def __step_matches(
        self,
        step: int) -> bool:
        return self.global_step % step == 0
    
    def __register_layer_stats_hooks(self) -> None:
        # Create output stats dataframe.
        cols = {
            'epoch': int,
            'step': int,
            'module': str,
            'module-type': str,
            'shape': str,
            'metric': str,
            'value': float,
        }
        self.__output_stats = pd.DataFrame(columns=cols.keys())
        self.__gradient_stats = pd.DataFrame(columns=cols.keys())

        # Only register hooks for leaf layers (no submodules).
        df = layer_summary(self.__network, params_only=False, leafs_only=True)
        modules = df['module'].tolist()
        for n, m in self.__network.named_modules():
            if n in modules:
                m.register_forward_hook(self.__get_forward_hook(n))
                m.register_backward_hook(self.__get_backward_hook(n))

    def __get_forward_hook(
        self,
        name: str) -> Callable:
        def hook(module, _, output) -> None:
            # Save output stats.
            if module.training and self.__interval_matches(self.__layer_stats_record_interval):
                metrics = [
                    'output-min',
                    'output-max',
                    'output-mean',
                    'output-std',
                ]
                values = [
                    output.min().item(),
                    output.max().item(),
                    output.mean().item(),
                    output.std().item(),
                ]
                for m, v, in zip(metrics, values):
                    data = {
                        'epoch': self.current_epoch,
                        'step': self.global_step,
                        'module': name,
                        'module-type': module.__class__.__name__,
                        'shape': str(list(output.shape)),
                        'metric': m,
                        'value': v,
                    }
                    self.__output_stats = append_row(self.__output_stats, data)

        return hook

    def __get_backward_hook(
        self,
        name: str) -> Callable:
        # 'grad_input' is not straightforward - https://discuss.pytorch.org/t/exact-meaning-of-grad-input-and-grad-output/14186/7.
        def hook(module, _, grad_output) -> None:
            # 'grad_output' is a tuple because some layers (e.g. torch.split) return multiple outputs.
            assert len(grad_output) == 1
            grad_output = grad_output[0]

            # Save gradient stats.
            if module.training and self.__interval_matches(self.__layer_stats_record_interval):
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
                        'module': name,
                        'module-type': module.__class__.__name__,
                        'shape': str(list(grad_output.shape)),
                        'metric': m,
                        'value': v,
                    }
                    self.__gradient_stats = append_row(self.__gradient_stats, data)

        return hook
