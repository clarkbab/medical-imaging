import csv
import numpy as np
import os
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR, ReduceLROnPlateau
from typing import Dict, List, Literal, Optional, OrderedDict, Tuple, Union

from mymi import config
from mymi import logging
from mymi.losses import DiceWithFocalLoss
from mymi.metrics import ncc
from mymi.models import replace_ckpt_alias
from mymi.models.architectures import MultiUNet3D
from mymi.transforms import apply_dvf
from mymi.typing import ModelName
from mymi.utils import gpu_usage_nvml

class RegSegModel(pl.LightningModule):
    def __init__(
        self,
        cyclic_min: Optional[float] = None,
        cyclic_max: Optional[float] = None,
        log_on_epoch: bool = True,
        log_on_step: bool = False,
        loss: nn.Module = DiceWithFocalLoss(),
        lr_find: bool = False,
        lr_init: float = 1e-3,
        metrics: List[str] = [],
        model_name: str = 'model-name',
        model_type: str = 'reg',
        run_name: str = 'run-name',
        use_lr_scheduler: bool = False,
        use_weights: bool = False,
        weight_decay: float = 0,
        weights: Optional[Union[List[float], List[List[float]]]] = None,
        weights_schedule: Optional[List[int]] = None,
        **kwargs) -> None:
        super().__init__()
        self.__cyclic_min = cyclic_min
        self.__cyclic_max = cyclic_max
        self.__log_on_epoch = log_on_epoch
        self.__log_on_step = log_on_step
        self.__loss = loss
        self.lr = lr_init        # 'self.lr' is default key that LR finder looks for on module.
        self.__lr_find = lr_find
        self.__metrics = metrics
        self.__mean_metrics = {}
        self.__model_name = model_name
        self.__model_type = model_type
        self.__name = None
        self.__run_name = run_name
        self.__use_lr_scheduler = use_lr_scheduler
        self.__weight_decay = weight_decay

        # Set network architecture.
        if model_type == 'reg':
            self.__n_input_channels = 2     # Pre/mid-treatment CT scans.
            self.__n_output_channels = 3    # 3-dimensional deformable vector field (DVF).
        else:
            raise ValueError(f"Invalid 'model_type' '{model_type}'.")
        self.__network = MultiUNet3D(self.__n_output_channels, n_input_channels=self.__n_input_channels, use_softmax=False, **kwargs)

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
                raise ValueError(f"Can't load RegSegModel '{model}', has only completed {n_epochs_complete} of {n_epochs} epochs.")

        # Load model.
        model = replace_ckpt_alias(model)
        filepath = os.path.join(config.directories.models, model[0], model[1], f"{model[2]}.ckpt")
        if not os.path.exists(filepath):
            raise ValueError(f"RegSegModel '{model}' not found.")

        # Load checkpoint.
        segmenter = RegSegModel.load_from_checkpoint(filepath, cw_cvg_calculate=cw_cvg_calculate, **kwargs)
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

            opt['lr_scheduler'] = CyclicLR(self.__optimiser, self.__cyclic_min, self.__cyclic_max, cycle_momentum=False)
            # opt['lr_scheduler'] = MultiStepLR(self.__optimiser, self.__lr_milestones, gamma=0.1)
            # opt['lr_scheduler'] = ReduceLROnPlateau(self.__optimiser, factor=0.5, patience=200, verbose=True)

            logging.info(f"Using cyclic LR with min={self.__cyclic_min}, max={self.__cyclic_max}).")

        return opt

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        return self.__network(x)

    def training_step(self, batch, batch_idx):
        # Forward pass.
        desc, fixed_input, moving_input, fixed_label, moving_label, fixed_mask, moving_mask, weights = batch
        batch_size = len(desc)
        assert batch_size == 1

        # Handle different model types.
        if self.__model_type == 'reg':
            input = torch.cat((fixed_input, moving_input), dim=1)
            dvf = self.forward(input)
            y_hat = apply_dvf(moving_input, dvf)
            loss = self.__loss(y_hat, fixed_input)

        # Report loss.
        if np.isnan(loss.item()):
            print(desc)
        else:
            self.log('train/loss', loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Report metrics.
        fixed_input = fixed_input.cpu().numpy()
        y_hat = y_hat.detach().cpu().numpy()
        if 'ncc' in self.__metrics:
            ncc_val = ncc(y_hat, fixed_input)
            self.log(f'train/ncc', ncc_val, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        return loss

    def validation_step(self, batch, batch_idx):
        desc, fixed_input, moving_input, fixed_label, moving_label, fixed_mask, moving_mask, weights = batch
        batch_size = len(desc)
        assert batch_size == 1

        # Forward pass.
        fixed_input = fixed_input.unsqueeze(1)
        moving_input = moving_input.unsqueeze(1)
        input = torch.cat((fixed_input, moving_input), dim=1)
        input = input.type(torch.float32)       # Why do we need this???
        dvf = self.forward(input)
        y_hat = apply_dvf(moving_input, dvf)

        # Calculate loss.
        loss = self.__loss(y_hat, fixed_input)
        self.log('val/loss', loss, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Record gpu usage.
        for i, usage_mb in enumerate(gpu_usage_nvml()):
            self.log(f'gpu/{i}', usage_mb, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Report metrics.
        fixed_input = fixed_input.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        if 'ncc' in self.__metrics:
            ncc_val = ncc(y_hat, fixed_input)
            self.log(f'val/ncc', ncc_val, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

            # Save metrics to calculate mean across validation set. Used to track convergence.
            if 'val/ncc' not in self.__mean_metrics:
                self.__mean_metrics['val/ncc'] = []
            self.__mean_metrics['val/ncc'] += [ncc_val]

    def on_validation_epoch_end(self):
        if 'ncc' in self.__metrics:
            mean_ncc = np.mean(self.__mean_metrics['val/ncc'])
            self.log(f'val/internal/ncc', mean_ncc, on_epoch=self.__log_on_epoch, on_step=self.__log_on_step)

        # Reset mean metrics.
        self.__mean_metrics = {}
