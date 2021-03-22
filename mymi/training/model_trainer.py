from datetime import datetime
import logging
import numpy as np
import os
import torch
from torch.autograd import profiler
from torch.cuda.amp import autocast, GradScaler
# from torch.autograd.profiler import profile, tensorboard_trace_handler
from torch.autograd.profiler import profile
from torch.utils.tensorboard import SummaryWriter

from mymi import loaders
from mymi import plotter
from mymi import utils
from mymi.metrics import dice as dice_metric

data_dir = os.environ['MYMI_DATA']
TENSORBOARD_DIR = os.path.join(data_dir, 'tensorboard')
CHECKPOINT_DIR = os.path.join(data_dir, 'checkpoints')

class ModelTrainer:
    def __init__(self, train_loader, validation_loader, optimiser, loss_fn, visual_loader, 
        max_epochs=100, run_name=None, metrics=('dice'), device=torch.device('cpu'), print_interval='epoch', 
        record_interval='epoch', validation_interval='epoch', print_format='.10f', is_reporter=False,
        mixed_precision=False, log_info=logging.info):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.visual_loader = visual_loader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.device = device
        self.print_interval = print_interval
        self.record_interval = record_interval
        self.validation_interval = validation_interval
        self.print_format = print_format
        self.best_loss = None
        self.max_epochs_without_improvement = 5
        self.num_epochs_without_improvement = 0
        self.run_name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') if run_name is None else run_name
        self.is_reporter = is_reporter
        self.log_info = log_info
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler(enabled=mixed_precision)
        if is_reporter:
            self.writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, self.run_name))

            # Add hyperparameters.
            hparams = {
                'run-name': self.run_name,
                'loss-function': str(self.loss_fn),
                'max-epochs': str(self.max_epochs),
                'mixed-precision': str(self.mixed_precision),
                'optimiser': str(self.optimiser),
                'transform': str(self.train_loader.dataset.transform),
            }
            self.writer.add_hparams(hparams, {}, run_name='hparams')

        self.running_scores = {
            'print': {
                'loss': 0
            },
            'record': {
                'loss': 0
            },
            'validation-print': {
                'loss': 0
            },
            'validation-record': {
                'loss': 0
            }
        }
        if 'dice' in self.metrics: 
            self.running_scores['print']['dice'] = 0
            self.running_scores['record']['dice'] = 0
            self.running_scores['validation-print']['dice'] = 0
            self.running_scores['validation-record']['dice'] = 0

    def __call__(self, model):
        """
        effect: updates the model parameters.
        model: the model to train.
        """
        # Put in training mode.
        model.train()

        for epoch in range(self.max_epochs):
            for batch, (input, label) in enumerate(self.train_loader):
                # Convert input and label.
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self.device), label.to(self.device)

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

                # Calculate other metrics.
                if 'dice' in self.metrics:
                    dice = dice_metric(pred, label)
                    self.running_scores['print']['dice'] += dice.item()
                    self.running_scores['record']['dice'] += dice.item()

                # Record training info to Tensorboard.
                iteration = epoch * len(self.train_loader) + batch
                if self.is_reporter and self.should_record(iteration):
                    self.record_training_results(iteration)
                    self.reset_record_scores()
                
                # Print results.
                if self.should_print(iteration, len(self.train_loader)):
                    self.print_training_results(epoch, iteration)
                    self.reset_print_scores()

                # Perform validation and checkpointing.
                if self.is_reporter and self.should_validate(iteration):
                    self.validate_model(model, epoch, iteration)

                # Check early stopping.
                if self.num_epochs_without_improvement >= self.max_epochs_without_improvement:
                    self.log_info(f"Stopping early due to {self.num_epochs_without_improvement} epochs without improved validation score.")
                    return

        self.log_info(f"Maximum epochs ({self.max_epochs} reached.")

    def validate_model(self, model, epoch, iteration):
        model.eval()

        # Plot validation images for visual indication of improvement.
        for batch, (input, label) in enumerate(self.visual_loader):
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            with autocast(enabled=self.mixed_precision):
                pred = model(input)

            # For each of the views, plot a batch of images.
            views = ('sagittal', 'coronal', 'axial')
            for view in views:
                # Get the centroid plane for each image in batch.
                centroids = self.get_batch_centroids(label, view)

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
                dice = dice_metric(pred, label)
                self.running_scores['validation-record']['dice'] += dice.item()
                self.running_scores['validation-print']['dice'] += dice.item()

            # Print results.
            if self.should_print(val_batch, len(self.validation_loader)):
                self.print_validation_results(epoch, val_batch)
                self.reset_validation_print_scores()
        
        # Record validation results on Tensorboard.
        self.record_validation_results(iteration)
        self.reset_validation_record_scores()

        # Check for validation improvement.
        if self.best_loss is None or loss < self.best_loss:
            self.save_model(model, iteration, loss)
            self.num_epochs_without_improvement = 0
        else:
            self.num_epochs_without_improvement += 1

        model.train()

    def save_model(self, model, iteration, loss):
        self.log_info(f"Saving model at iteration {iteration}, achieved best loss: {loss:{self.print_format}}")
        filepath = os.path.join(CHECKPOINT_DIR, self.run_name, 'best.pt')
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
        message = f"[{epoch}, {batch}] Loss: {loss:{self.print_format}}"

        if 'dice' in self.metrics:
            dice = self.running_scores['print']['dice'] / print_interval
            message += f", Dice: {dice:{self.print_format}}"

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
        message = f"Validation - [{epoch}, {batch}] Loss: {loss:{self.print_format}}"

        if 'dice' in self.metrics:
            dice = self.running_scores['validation-print']['dice'] / print_interval
            message += f", Dice: {dice:{self.print_format}}"

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

    def reset_print_scores(self):
        self.running_scores['print']['loss'] = 0
        self.running_scores['print']['dice'] = 0
    
    def reset_record_scores(self):
        self.running_scores['record']['loss'] = 0
        self.running_scores['record']['dice'] = 0

    def reset_validation_print_scores(self):
        self.running_scores['validation-print']['loss'] = 0
        self.running_scores['validation-print']['dice'] = 0
        
    def reset_validation_record_scores(self):
        self.running_scores['validation-record']['loss'] = 0
        self.running_scores['validation-record']['dice'] = 0

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
