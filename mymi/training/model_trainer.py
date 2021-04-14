import logging
import numpy as np
import os
import torch
from torch.autograd import profiler
from torch.cuda.amp import autocast, GradScaler
# from torch.autograd.profiler import profile, tensorboard_trace_handler
from torch.autograd.profiler import profile
from torch.utils.tensorboard import SummaryWriter

from mymi import checkpoint
from mymi import config
from mymi import loaders
from mymi import plotter
from mymi import utils
from mymi.metrics import batch_dice, batch_hausdorff_distance

PRINT_DP = '.10f'

class ModelTrainer:
    def __init__(self, loss_fn, optimiser, run_name, train_loader, validation_loader, visual_validation_loader,
        device=torch.device('cpu'), early_stopping=False, is_reporter=False, log_info=logging.info, max_epochs=500,
        mixed_precision=True, metrics=('dice', 'hausdorff'), print_interval='epoch', record_interval='epoch',
        spacing=None, validation_interval='epoch'):
        """
        effect: sets the initial trainer values.
        args:
            loss_fn: objective function of the training.
            optimiser: updates the model parameters in response to gradients.
            run_name: the name of the run to show in reporting.
            train_loader: provides the training input and label batches.
            validation_loader: provides the validation input and label batches.
            visual_validation_loader: provides the visual validation input and label batches.
        kwargs:
            device: the device to train on.
            early_stopping: if the training should use early stopping or not.
            is_reporter: if this process should report or not.
            log_info: the logging function. Allows us to include multi-process info if required.
            max_epochs: the maximum number of epochs to run training.
            mixed_precision: run the training using PyTorch mixed precision training.
            metrics: the metrics to print and record during training.
            print_interval: how often to print results during training.
            record_interval: how often to record results during training.
            spacing: the voxel spacing. Required for calculating Hausdorff distance.
            validation_interval: how often to run the validation.
        """
        self.device = device
        self.early_stopping = early_stopping
        self.is_reporter = is_reporter
        if is_reporter:
            # Create tensorboard writer.
            self.writer = SummaryWriter(os.path.join(config.tensorboard_dir, run_name))

            # Add hyperparameters.
            hparams = {
                'run-name': run_name,
                'loss-function': str(loss_fn),
                'max-epochs': max_epochs,
                'mixed-precision': mixed_precision,
                'optimiser': str(optimiser),
                'transform': str(train_loader.dataset.transform),
            }
            self.writer.add_hparams(hparams, {}, run_name='hparams')
        self.log_info = log_info
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.max_epochs_since_improvement = 20
        self.metrics = metrics
        self.min_validation_loss = np.inf
        self.mixed_precision = mixed_precision
        self.num_epochs_since_improvement = 0
        self.optimiser = optimiser
        self.print_interval = print_interval
        self.record_interval = record_interval
        self.run_name = run_name
        self.scaler = GradScaler(enabled=mixed_precision)
        self.spacing = spacing
        if 'hausdorff' in metrics:
            assert spacing is not None, 'Voxel spacing must be provided when calculating Hausdorff distance.'
        self.train_loader = train_loader
        self.validation_interval = validation_interval
        self.validation_loader = validation_loader
        self.visual_validation_loader = visual_validation_loader

        # Initialise running scores.
        self.running_scores = {}
        keys = ['print', 'record', 'validation-print', 'validation-record']
        for key in keys:
            self.running_scores[key] = {}
            self.reset_running_scores(key)

    def __call__(self, model):
        """
        effect: updates the model parameters.
        model: the model to train.
        """
        # Put in training mode.
        model.train()

        for epoch in range(self.max_epochs):
            for batch, (input, label) in enumerate(self.train_loader):
                iteration = epoch * len(self.train_loader) + batch
                # Convert input and label.
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self.device), label.to(self.device)
                print(batch)

                # Add model structure.
                if self.is_reporter and epoch == 0 and batch == 0:
                    # Error when adding graph with 'mixed-precision' training.
                    if not self.mixed_precision:
                        self.writer.add_graph(model, input)

                # Perform forward/backward pass.
                with autocast(enabled=self.mixed_precision):
                    pred = model(input)
                    loss = self.loss_fn(pred, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad()
                self.running_scores['print']['loss'] += loss.item()
                self.running_scores['record']['loss'] += loss.item()

                # Convert to binary prediction.
                pred = pred.argmax(axis=1)

                # Move data to CPU for metric calculations.
                pred, label = pred.cpu(), label.cpu()

                # Calculate other metrics.
                if 'dice' in self.metrics:
                    dice = batch_dice(pred, label)
                    self.running_scores['print']['dice'] += dice.item()
                    self.running_scores['record']['dice'] += dice.item()

                if 'hausdorff' in self.metrics:
                    print('starting hausdorff')
                    hausdorff = batch_hausdorff_distance(pred, label, spacing=self.spacing)
                    self.running_scores['print']['hausdorff'] += hausdorff.item()
                    self.running_scores['record']['hausdorff'] += hausdorff.item()
                    print('finished hausdorff')

                # Record training info to Tensorboard.
                if self.is_reporter and self.should_record(iteration):
                    self.record_training_results(iteration)
                    self.reset_running_scores('record')
                
                # Print results.
                if self.should_print(iteration, len(self.train_loader)):
                    self.print_training_results(epoch, iteration)
                    self.reset_running_scores('print')

                # Perform validation and checkpointing.
                if self.is_reporter and self.should_validate(iteration):
                    self.validate_model(model, epoch, iteration)

                # Check early stopping.
                if self.early_stopping:
                    if self.num_epochs_since_improvement >= self.max_epochs_since_improvement:
                        self.log_info(f"Stopping early due to {self.num_epochs_since_improvement} epochs without improved validation score.")
                        return

        self.log_info(f"Maximum epochs ({self.max_epochs} reached.")

    def validate_model(self, model, epoch, iteration):
        model.eval()

        # Plot validation images for visual indication of improvement.
        for batch, (input, label) in enumerate(self.visual_validation_loader):
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            with autocast(enabled=self.mixed_precision):
                pred = model(input)

            # Convert prediction to binary values.
            pred = pred.argmax(axis=1)

            # For each of the views, plot a batch of images.
            views = ('sagittal', 'coronal', 'axial')
            for view in views:
                # Get the centroid plane for each image in batch.
                centroids = utils.get_batch_centroids(label, view)

                # Get figure.
                figure = plotter.plot_batch(input, centroids, label=label, pred=pred, view=view, return_figure=True)

                # Write figure to tensorboard.
                tag = f"Validation - batch={batch}, view={view}"
                self.writer.add_figure(tag, figure, global_step=iteration)

        # Calculate validation score.
        for val_batch, (input, label) in enumerate(self.validation_loader):
            # Convert input data.
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            with autocast(enabled=self.mixed_precision):
                pred = model(input)
                loss = self.loss_fn(pred, label)
            self.running_scores['validation-record']['loss'] += loss.item()
            self.running_scores['validation-print']['loss'] += loss.item()

            if 'dice' in self.metrics:
                dice = batch_dice(pred, label)
                self.running_scores['validation-record']['dice'] += dice.item()
                self.running_scores['validation-print']['dice'] += dice.item()

            if 'hausdorff' in self.metrics:
                hausdorff = batch_hausdorff_distance(pred, label, spacing=self.spacing)
                self.running_scores['print']['hausdorff'] += hausdorff.item()
                self.running_scores['record']['hausdorff'] += hausdorff.item()

            # Print results.
            if self.should_print(val_batch, len(self.validation_loader)):
                self.print_validation_results(epoch, val_batch)
                self.reset_running_scores('validation-print')

        # Check for validation improvement.
        record_interval = len(self.validation_loader)
        loss = self.running_scores['validation-record']['loss'] / record_interval

        # Save model checkpoint.
        if loss < self.min_validation_loss:
            info = {
                'epoch': epoch,
                'iteration': iteration,
                'loss': loss
            }
            checkpoint.save(model, self.run_name, self.optimiser, info=info)
            self.min_validation_loss = loss
            self.num_epochs_since_improvement = 0
        else:
            self.num_epochs_since_improvement += 1
        
        # Record validation results on Tensorboard.
        self.record_validation_results(iteration)
        self.reset_running_scores('validation-record')

        model.train()

    def save_model(self, model, iteration, loss):
        self.log_info(f"Saving model at iteration {iteration}, achieved best loss: {loss:{PRINT_DP}}")
        filepath = os.path.join(config.checkpoint_dir, self.run_name, 'best.pt')
        info = {
            'iteration': iteration,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(info, filepath)
        
    def should_print(self, iteration, loader_length):
        if ((self.print_interval == 'epoch' and (iteration + 1) % len(self.train_loader) == 0) or
            (self.print_interval != 'epoch' and (iteration + 1) % self.print_interval == 0)):
            return True
        else:
            return False

    def should_record(self, iteration):
        if ((self.record_interval == 'epoch' and (iteration + 1) % len(self.train_loader) == 0) or
            (self.record_interval != 'epoch' and (iteration + 1) % self.record_interval == 0)):
            return True
        else:
            return False

    def should_validate(self, iteration):
        if ((self.validation_interval == 'epoch' and (iteration + 1) % len(self.train_loader) == 0) or
            (self.validation_interval != 'epoch' and (iteration + 1) % self.validation_interval == 0)):
            return True
        else:
            return False

    def print_training_results(self, epoch, batch):
        """
        effect: logs an update to STDOUT.
        """
        print_interval = len(self.train_loader) if self.print_interval == 'epoch' else self.print_interval
        loss = self.running_scores['print']['loss'] / print_interval
        message = f"[{epoch}, {batch}] Loss: {loss:{PRINT_DP}}"

        if 'dice' in self.metrics:
            dice = self.running_scores['print']['dice'] / print_interval
            message += f", Dice: {dice:{PRINT_DP}}"

        if 'hausdorff' in self.metrics:
            hausdorff = self.running_scores['print']['hausdorff'] / print_interval
            message += f", Hausdorff: {hausdorff:{PRINT_DP}}"

        self.log_info(message)
        
    def print_validation_results(self, epoch, batch):
        """
        effect: logs an update to STDOUT.
        args:
            epoch: the epoch we're up to.
            batch: the batch we're up to.
        """
        print_interval = len(self.validation_loader) if self.print_interval == 'epoch' else self.print_interval
        loss = self.running_scores['validation-print']['loss'] / print_interval
        message = f"Validation - [{epoch}, {batch}] Loss: {loss:{PRINT_DP}}"

        if 'dice' in self.metrics:
            dice = self.running_scores['validation-print']['dice'] / print_interval
            message += f", Dice: {dice:{PRINT_DP}}"

        if 'hausdorff' in self.metrics:
            hausdorff = self.running_scores['validation-print']['hausdorff'] / print_interval
            message += f", Hausdorff: {hausdorff:{PRINT_DP}}"

        self.log_info(message)

    def record_training_results(self, iteration):
        """
        effect: sends training results to Tensorboard.
        args:
            epoch: the epoch we're up to.
            iteration: the iteration we're up to.
        """
        record_interval = len(self.train_loader) if self.record_interval == 'epoch' else self.record_interval
        loss = self.running_scores['record']['loss'] / record_interval
        self.writer.add_scalar('Loss/train', loss, iteration)
        
        if 'dice' in self.metrics:
            dice = self.running_scores['record']['dice'] / record_interval
            self.writer.add_scalar('Dice/train', dice, iteration)

        if 'hausdorff' in self.metrics:
            hausdorff = self.running_scores['record']['hausdorff'] / record_interval
            self.writer.add_scalar('Hausdorff/train', hausdorff, iteration)

    def record_validation_results(self, iteration):
        """
        effect: sends validation results to Tensorboard.
        """
        record_interval = len(self.validation_loader)
        loss = self.running_scores['validation-record']['loss'] / record_interval
        self.writer.add_scalar('Loss/validation', loss, iteration)

        if 'dice' in self.metrics:
            dice = self.running_scores['validation-record']['dice'] / record_interval
            self.writer.add_scalar('Dice/validation', dice, iteration)

        if 'hausdorff' in self.metrics:
            hausdorff = self.running_scores['record']['hausdorff'] / record_interval
            self.writer.add_scalar('Hausdorff/train', hausdorff, iteration)

    def reset_running_scores(self, key):
        self.running_scores[key]['loss'] = 0
        if 'dice' in self.metrics:
            self.running_scores[key]['dice'] = 0
        if 'hausdorff' in self.metrics:
            self.running_scores[key]['hausdorff'] = 0

    def get_batch_centroids(self, label_b, plane):
        """
        returns: the centroid location of the label along the plane axis, for each
            image in the batch.
        args:
            label_b: the batch of labels.
            plane: the plane along which to find the centroid.
        """
        assert plane in ('axial', 'coronal', 'sagittal')

        # Move data to CPU.
        label_b = label_b.cpu()

        # Determine axes to sum over.
        if plane == 'axial':
            axes = (0, 1)
        elif plane == 'coronal':
            axes = (0, 2)
        elif plane == 'sagittal':
            axes = (1, 2)

        centroids = np.array([], dtype=np.int)

        # Loop through batch and get centroid for each label.
        for label_i in label_b:
            # Get weighting along 'plane' axis.
            weights = label_i.sum(axes)

            # Get average weighted sum.
            indices = np.arange(len(weights))
            avg_weighted_sum = (weights * indices).sum() /  weights.sum()

            # Get centroid index.
            centroid = np.round(avg_weighted_sum).long()
            centroids = np.append(centroids, centroid)

        return centroids
