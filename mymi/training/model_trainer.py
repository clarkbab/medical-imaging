from collections.abc import Callable, Iterable
import logging
import numpy as np
import os
import torch
from torch.autograd import profiler
from torch.cuda.amp import autocast, GradScaler
# from torch.autograd.profiler import profile, tensorboard_trace_handler
from torch.autograd.profiler import profile
from torch.utils.tensorboard import SummaryWriter
from typing import *

from mymi import checkpoint
from mymi import config
from mymi import loaders
from mymi.postprocessing import batch_largest_connected_component
from mymi.reporting import WandbReporter
from mymi import utils
from mymi.metrics import batch_dice, sitk_batch_hausdorff_distance

PRINT_DP = '.10f'

class ModelTrainer:
    def __init__(
        self,
        loss_fn: torch.nn.Module,
        model_name: str,
        optimiser: torch.optim.Optimizer,
        run_name: str,
        train_loader: torch.utils.data.DataLoader,
        validate_loader: torch.utils.data.DataLoader,
        validate_loader_visual: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu'),
        early_stopping: bool = False,
        hausdorff_delay: int = 500,
        log_info: Callable[[str], None] = logging.info,
        max_epochs: int = 500,
        mixed_precision: bool = True,
        metrics: Iterable[str] = ('dice', 'hausdorff'),
        train_print_interval: Union[str, int] = 'epoch',
        train_report_interval: Union[str, int] = 'epoch',
        report: bool = True,
        report_offline: bool = False,
        spacing: Optional[Iterable[float]] = None,
        validate: bool = True,
        validate_interval: Union[str, int] ='epoch',
        validate_print_interval: Union[str, int] = 'epoch',
        validate_report_interval: Union[str, int] = 'epoch') -> None:
        """
        effect: sets the initial trainer values.
        args:
            loss_fn: objective function of the training.
            model_name: the name of the model, e.g. 'parotid-left-3d-localiser'.
            optimiser: updates the model parameters in response to gradients.
            run_name: the name of the run to show in reporting.
            train_loader: provides the training input and label batches.
            validate_loader: provides the validation input and label batches.
            validate_loader_visual: provides the visual validation input and label batches.
        kwargs:
            device: the device to train on.
            early_stopping: if the training should use early stopping or not.
            hausdorff_delay: calculate HD after this many steps due to HD expense.
            log_info: the logging function. Allows us to include multi-process info if required.
            max_epochs: the maximum number of epochs to run training.
            mixed_precision: run the training using PyTorch mixed precision training.
            metrics: the metrics to print and report during training.
            train_print_interval: how often to print results during training.
            train_report_interval: how often to report results during training.
            report: turns reporting on and off.
            report_offline: creates reporting files locally, to be uploaded later.
            spacing: the voxel spacing. Required for calculating Hausdorff distance.
            validate: turns validation on and off.
            validate_interval: how often to run the validation.
            validate_print_interval: how often to print results during validation.
            validate_report_interval: how often to report results during validation.
        """
        self._device = device
        self._early_stopping = early_stopping
        self._hausdorff_delay = hausdorff_delay
        if report:
            # Create tensorboard writer.
            self._reporter = WandbReporter(model_name, run_name, offline=report_offline)

            # Add hyperparameters.
            hparams = {
                'run-name': run_name,
                'loss-function': str(loss_fn),
                'max-epochs': max_epochs,
                'mixed-precision': mixed_precision,
                'optimiser': str(optimiser),
                'transform': str(train_loader.dataset._transform),
            }
            self._reporter.add_hyperparameters(hparams)
        self._log_info = log_info
        self._loss_fn = loss_fn
        self._max_epochs = max_epochs
        self._max_epochs_since_improvement = 20
        self._metrics = metrics
        self._min_validate_loss = np.inf
        self._mixed_precision = mixed_precision
        self._model_name = model_name
        self._num_epochs_since_improvement = 0
        self._optimiser = optimiser
        self._report = report
        self._run_name = run_name
        self._scaler = GradScaler(enabled=mixed_precision)
        self._spacing = spacing
        if 'hausdorff' in metrics:
            assert spacing is not None, 'Voxel spacing must be provided when calculating Hausdorff distance.'
        self._train_loader = train_loader
        self._train_print_interval = len(train_loader) if train_print_interval == 'epoch' else train_print_interval
        self._train_report_interval = len(train_loader) if train_report_interval == 'epoch' else train_report_interval
        self._validate = validate
        self._validate_interval = len(train_loader) if validate_interval == 'epoch' else validate_interval
        self._validate_loader = validate_loader
        self._validate_loader_visual = validate_loader_visual
        self._validate_print_interval = len(validate_loader) if validate_print_interval == 'epoch' else validate_print_interval
        self._validate_report_interval = len(validate_loader) if validate_report_interval == 'epoch' else validate_report_interval

        # Initialise running scores.
        self._running_scores = {}
        keys = ['print', 'report', 'validate-print', 'validate-report', 'validate-checkpoint']
        for key in keys:
            self._running_scores[key] = {}
            self._reset_running_scores(key)

    def __call__(
        self,
        model: torch.nn.Module) -> None:
        """
        effect: performs training to update model parameters whilst validating model performance.
        args:
            model: the model to train.
        """
        # Put in training mode.
        model.train()

        for epoch in range(self._max_epochs):
            for batch, (input, label) in enumerate(self._train_loader):
                # Calculate training step.
                step = epoch * len(self._train_loader) + batch

                # Convert input and label.
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self._device), label.to(self._device)

                # Add model structure.
                if self._report and epoch == 0 and batch == 0:
                    # Error when adding graph with 'mixed-precision' training.
                    if not self._mixed_precision:
                        self._reporter.add_model_graph(model, input)

                # Perform forward/backward pass.
                with autocast(enabled=self._mixed_precision):
                    pred = model(input)
                    loss = self._loss_fn(pred, label)
                self._scaler.scale(loss).backward()
                self._scaler.step(self._optimiser)
                self._scaler.update()
                self._optimiser.zero_grad()
                self._running_scores['print']['loss'] += [loss.item()]
                self._running_scores['report']['loss'] += [loss.item()]

                # Convert to binary prediction.
                pred = pred.argmax(axis=1)

                # Move data to CPU for metric calculations.
                pred, label = pred.cpu(), label.cpu()

                # Calculate other metrics.
                if 'dice' in self._metrics:
                    dice = batch_dice(pred, label)
                    self._running_scores['print']['dice'] += [dice.item()]
                    self._running_scores['report']['dice'] += [dice.item()]

                if 'hausdorff' in self._metrics and step > self._hausdorff_delay:
                    # Can't calculate HD if prediction is empty.
                    if pred.sum() > 0:
                        hausdorff = sitk_batch_hausdorff_distance(pred, label, spacing=self._spacing)
                        self._running_scores['print']['hausdorff'] += [hausdorff.item()]
                        self._running_scores['report']['hausdorff'] += [hausdorff.item()]
                
                # Print results.
                if self._is_print_step(self._train_print_interval, step):
                    self._print_training_results(epoch, batch, step)
                    self._reset_running_scores('print')

                # Report results.
                if self._report and self._is_report_step(self._train_report_interval, step):
                    self._report_training_results(step)
                    self._reset_running_scores('report')

                # Perform validation and checkpointing.
                if self._validate and self._is_validate_step(step):
                    self._validate_model(model, epoch, batch, step)

                # Check early stopping.
                if self._early_stopping:
                    if self._num_epochs_since_improvement >= self._max_epochs_since_improvement:
                        self._log_info(f"Stopping early due to {self._num_epochs_since_improvement} epochs without improved validation score.")
                        return

        self._log_info(f"Maximum epochs ({self._max_epochs} reached.")

    def _validate_model(
        self,
        model: torch.nn.Module,
        train_epoch: int,
        train_batch: int,
        train_step: int) -> None:
        """
        effect: evaluates the model on the validation and visual validation sets.
        args:
            model: the model to evaluate.
            train_epoch: the current training epoch.
            train_batch: the current training batch.
            train_step: the current training step.
        """
        model.eval()

        # Calculate validation score.
        for step, (input, label) in enumerate(self._validate_loader):
            # Convert input data.
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self._device), label.to(self._device)

            # Perform forward pass.
            with autocast(enabled=self._mixed_precision):
                pred = model(input)
                loss = self._loss_fn(pred, label)
            self._running_scores['validate-checkpoint']['loss'] += [loss.item()]
            self._running_scores['validate-print']['loss'] += [loss.item()]
            self._running_scores['validate-report']['loss'] += [loss.item()]

            # Convert to binary prediction.
            pred = pred.argmax(axis=1)

            # Move data to CPU for metric calculations.
            pred, label = pred.cpu(), label.cpu()

            if 'dice' in self._metrics:
                dice = batch_dice(pred, label)
                self._running_scores['validate-print']['dice'] += [dice.item()]
                self._running_scores['validate-report']['dice'] += [dice.item()]

            if 'hausdorff' in self._metrics and train_step > self._hausdorff_delay:
                # Can't calculate HD if prediction is empty.
                if pred.sum() > 0:
                    hausdorff = sitk_batch_hausdorff_distance(pred, label, spacing=self._spacing)
                    self._running_scores['validate-print']['hausdorff'] += [hausdorff.item()]
                    self._running_scores['validate-report']['hausdorff'] += [hausdorff.item()]

            # Print results.
            if self._is_print_step(self._validate_print_interval, step):
                self._print_validate_results(train_epoch, train_batch, train_step, step)
                self._reset_running_scores('validate-print')

            # Report results.
            if self._report and self._is_report_step(self._validate_report_interval, step):
                self._report_validate_results(train_step)
                self._reset_running_scores('validate-report')

        # Check for validation loss improvement.
        loss = np.mean(self._running_scores['validate-checkpoint']['loss'])
        if loss < self._min_validate_loss:
            # Save model checkpoint.
            info = {
                'train-epoch': train_epoch,
                'train-batch': train_batch,
                'train-step': train_step,
                'validate-loss': loss
            }
            checkpoint.save(model, self._model_name, self._optimiser, self._run_name, info=info)
            self._min_validate_loss = loss
            self._num_epochs_since_improvement = 0
        else:
            self._num_epochs_since_improvement += 1
        self._reset_running_scores('validate-checkpoint')

        # Plot validation images for visual indication of improvement.
        if self._report:
            for step, (input, label) in enumerate(self._validate_loader_visual):
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self._device), label.to(self._device)

                # Perform forward pass.
                with autocast(enabled=self._mixed_precision):
                    pred = model(input)

                # Convert prediction to binary values.
                pred = pred.argmax(axis=1)

                # Move data to CPU for calculations.
                input, pred, label = input.squeeze(1).cpu(), pred.cpu(), label.cpu()

                # Loop through batch.
                for sample_idx in range(len(pred)):
                    # Load the label centroid.
                    label_centroid = np.round(np.argwhere(label[sample_idx] == 1).sum(1) / label[sample_idx].sum()).long()

                    # Report centroid slices for each view. 
                    for j, c in enumerate(label_centroid):
                        # Create index.
                        index = [slice(None), slice(None), slice(None)]
                        index[j] = c.item()

                        # Add figure.
                        class_labels = { 1: 'Parotid-Left' }
                        input_data = input[sample_idx][index]
                        label_data = label[sample_idx][index]
                        pred_data = pred[sample_idx][index]

                        # Rotate data and get axis name.
                        if j == 0:
                            axis = 'sagittal'
                            input_data = input_data.rot90()
                            label_data = label_data.rot90()
                            pred_data = pred_data.rot90()
                        elif j == 1:
                            axis = 'coronal'
                            input_data = input_data.rot90()
                            label_data = label_data.rot90()
                            pred_data = pred_data.rot90()
                        elif j == 2:
                            axis = 'axial'
                            input_data = input_data.rot90(-1)
                            label_data = label_data.rot90(-1)
                            pred_data = pred_data.rot90(-1)

                        self._reporter.add_figure(input_data, label_data, pred_data, train_step, step, sample_idx, axis, class_labels)

        model.train()
        
    def _is_print_step(
        self,
        interval: int,
        step: int) -> bool:
        """
        returns: whether the training or validation score should be printed.
        args:
            interval: the interval between printing.
            step: the current training or validation step.
        """
        if (step + 1) % interval == 0:
            return True
        else:
            return False

    def _is_report_step(
        self,
        interval: int,
        step: int) -> bool:
        """
        returns: whether the training score should be reported.
        args:
            interval: the interval between printing.
            step: the current training step.
        """
        if (step + 1) % interval == 0:
            return True
        else:
            return False

    def _is_validate_step(
        self,
        step: int) -> bool:
        """
        returns: whether the validation should be performed.
        args:
            step: the current training step.
        """
        if (step + 1) % self._validate_interval == 0:
            return True
        else:
            return False

    def _print_training_results(
        self,
        epoch: int,
        batch: int,
        step: int) -> None:
        """
        effect: logs averaged training results over the last print interval.
        args:
            epoch: the current training epoch.
            batch: the current training batch.
            step: the current training step.
        """
        # Get average training loss.
        mean_loss = np.mean(self._running_scores['print']['loss'])
        message = f"[E:{epoch}, B:{batch}, I:{step}] Loss: {mean_loss:{PRINT_DP}}"

        # Get additional metrics.
        if 'dice' in self._metrics:
            mean_dice = np.mean(self._running_scores['print']['dice'])
            message += f", DSC: {mean_dice:{PRINT_DP}}"

        if 'hausdorff' in self._metrics and step > self._hausdorff_delay:
            mean_hausdorff = np.mean(self._running_scores['print']['hausdorff'])
            message += f", HD: {mean_hausdorff:{PRINT_DP}}"

        self._log_info(message)
        
    def _print_validate_results(
        self,
        train_epoch: int,
        train_batch: int,
        train_step: int,
        validate_batch: int) -> None:
        """
        effect: logs the averaged validation results.
        args:
            train_epoch: the current training epoch.
            train_batch: the current training batch.
            train_step: the current training step.
            validate_batch: the current validation batch.
        """
        # Get average validation loss.
        mean_loss = np.mean(self._running_scores['validate-print']['loss'])
        message = f"VAL - [E:{train_epoch}, B:{train_batch}, I:{train_step}, VB:{validate_batch}] Loss: {mean_loss:{PRINT_DP}}"

        # Get additional metrics.
        if 'dice' in self._metrics:
            mean_dice = np.mean(self._running_scores['validate-print']['dice'])
            message += f", DSC: {mean_dice:{PRINT_DP}}"

        if 'hausdorff' in self._metrics and train_step > self._hausdorff_delay:
            mean_hausdorff = np.mean(self._running_scores['validate-print']['hausdorff'])
            message += f", HD: {mean_hausdorff:{PRINT_DP}}"

        self._log_info(message)

    def _report_training_results(
        self,
        step: int) -> None:
        """
        effect: reports averaged training results.
        args:
            step: the current training step.
        """
        mean_loss = np.mean(self._running_scores['report']['loss'])
        self._reporter.add_metric('Loss/train', mean_loss, step)
        
        if 'dice' in self._metrics:
            mean_dice = np.mean(self._running_scores['report']['dice'])
            self._reporter.add_metric('Dice/train', mean_dice, step)

        if 'hausdorff' in self._metrics and step > self._hausdorff_delay:
            mean_hausdorff = np.mean(self._running_scores['report']['hausdorff'])
            self._reporter.add_metric('Hausdorff/train', mean_hausdorff, step)

    def _report_validate_results(
        self,
        step: int) -> None:
        """
        effect: reports averaged validation results.
        args:
            step: the current training step. 
        """
        mean_loss = np.mean(self._running_scores['validate-report']['loss'])
        self._reporter.add_metric('Loss/validate', mean_loss, step)

        if 'dice' in self._metrics:
            mean_dice = np.mean(self._running_scores['validate-report']['dice'])
            self._reporter.add_metric('Dice/validate', mean_dice, step)

        if 'hausdorff' in self._metrics and step > self._hausdorff_delay:
            mean_hausdorff = np.mean(self._running_scores['validate-report']['hausdorff'])
            self._reporter.add_metric('Hausdorff/validate', mean_hausdorff, step)

    def _reset_running_scores(
        self,
        key: str) -> None:
        """
        effect: initialises the metrics under the key namespace.
        args:
            key: the metric namespace, e.g. print, report, etc.
        """
        self._running_scores[key]['loss'] = []
        if 'dice' in self._metrics:
            self._running_scores[key]['dice'] = []
        if 'hausdorff' in self._metrics:
            self._running_scores[key]['hausdorff'] = []
