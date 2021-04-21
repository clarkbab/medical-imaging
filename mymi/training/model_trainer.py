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
from mymi.postprocessing import batch_largest_connected_component
from mymi import utils
from mymi.metrics import batch_dice, sitk_batch_hausdorff_distance

PRINT_DP = '.10f'

class ModelTrainer:
    def __init__(self, loss_fn, optimiser, run_name, train_loader, validation_loader, visual_validation_loader,
        device=torch.device('cpu'), early_stopping=False, hausdorff_delay=200, is_primary=False, log_info=logging.info,
        max_epochs=500, mixed_precision=True, metrics=('dice', 'hausdorff'), print_interval='epoch', record=True,
        record_interval='epoch', spacing=None, validation_interval='epoch'):
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
            hausdorff_delay: calculate HD after this many iterations due to HD expense.
            is_primary: is this process the primary in the pool, i.e. responsible for validation and reporting.
            log_info: the logging function. Allows us to include multi-process info if required.
            max_epochs: the maximum number of epochs to run training.
            mixed_precision: run the training using PyTorch mixed precision training.
            metrics: the metrics to print and record during training.
            print_interval: how often to print results during training.
            record: turns recording on and off.
            record_interval: how often to record results during training.
            spacing: the voxel spacing. Required for calculating Hausdorff distance.
            validation_interval: how often to run the validation.
        """
        self.device = device
        self.early_stopping = early_stopping
        self.hausdorff_delay = hausdorff_delay
        self.is_primary = is_primary
        if is_primary and record:
            # Create tensorboard writer.
            self.recorder = SummaryWriter(os.path.join(config.directories.tensorboard, run_name))

            # Add hyperparameters.
            hparams = {
                'run-name': run_name,
                'loss-function': str(loss_fn),
                'max-epochs': max_epochs,
                'mixed-precision': mixed_precision,
                'optimiser': str(optimiser),
                'transform': str(train_loader.dataset.transform),
            }
            self.recorder.add_hparams(hparams, {}, run_name='hparams')
        self.log_info = log_info
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.max_epochs_since_improvement = 20
        self.metrics = metrics
        self.min_validation_loss = np.inf
        self.mixed_precision = mixed_precision
        self.num_epochs_since_improvement = 0
        self.optimiser = optimiser
        self.record = record
        self.run_name = run_name
        self.scaler = GradScaler(enabled=mixed_precision)
        self.spacing = spacing
        if 'hausdorff' in metrics:
            assert spacing is not None, 'Voxel spacing must be provided when calculating Hausdorff distance.'
        self.train_loader = train_loader
        self.train_print_interval = len(train_loader) if print_interval == 'epoch' else print_interval
        self.train_record_interval = len(train_loader) if record_interval == 'epoch' else record_interval
        self.validation_interval = len(train_loader) if validation_interval == 'epoch' else validation_interval
        self.validation_loader = validation_loader
        self.validation_print_interval = len(validation_loader) if print_interval == 'epoch' else print_interval
        self.validation_record_interval = len(validation_loader)
        self.visual_validation_loader = visual_validation_loader

        # Initialise running scores.
        self.running_scores = {}
        keys = ['print', 'record', 'validation-print', 'validation-record']
        for key in keys:
            self.running_scores[key] = {}
            self.reset_running_scores(key)

    def __call__(self, model):
        """
        effect: performs training to update model parameters whilst validating model performance.
        args:
            model: the model to train.
        """
        # Put in training mode.
        model.train()

        for epoch in range(self.max_epochs):
            for batch, (input, label) in enumerate(self.train_loader):
                # Calculate iteration.
                iteration = epoch * len(self.train_loader) + batch

                # Convert input and label.
                input, label = input.float(), label.long()
                input = input.unsqueeze(1)
                input, label = input.to(self.device), label.to(self.device)
                print(batch)

                # Add model structure.
                if self.is_primary and epoch == 0 and batch == 0:
                    # Error when adding graph with 'mixed-precision' training.
                    if not self.mixed_precision:
                        self.recorder.add_graph(model, input)

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

                if 'hausdorff' in self.metrics and iteration > self.hausdorff_delay:
                    # Get largest connected component in each prediction, this will speed up Hausdorff calculation.
                    # pred_cc = batch_largest_connected_component(pred)

                    print('starting hausdorff')
                    hausdorff = sitk_batch_hausdorff_distance(pred_cc, label, spacing=self.spacing)
                    self.running_scores['print']['hausdorff'] += hausdorff.item()
                    self.running_scores['record']['hausdorff'] += hausdorff.item()
                    print('finished hausdorff')

                # Record training info to Tensorboard.
                if self.is_primary and self.should_record(iteration):
                    self.record_training_results(iteration)
                    self.reset_running_scores('record')
                
                # Print results.
                if self.should_print(self.train_print_interval, iteration):
                    self.print_training_results(epoch, batch, iteration)
                    self.reset_running_scores('print')

                # Perform validation and checkpointing.
                if self.is_primary and self.should_validate(iteration):
                    self.validate_model(model, epoch, iteration)

                # Check early stopping.
                if self.early_stopping:
                    if self.num_epochs_since_improvement >= self.max_epochs_since_improvement:
                        self.log_info(f"Stopping early due to {self.num_epochs_since_improvement} epochs without improved validation score.")
                        return

        self.log_info(f"Maximum epochs ({self.max_epochs} reached.")

    def validate_model(self, model, epoch, iteration):
        """
        effect: evaluates the model on the validation and visual validation sets.
        args:
            model: the model to evaluate.
            epoch: the current training epoch.
            iteration: the current training iteration.
        """
        model.eval()

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

            # Convert to binary prediction.
            pred = pred.argmax(axis=1)

            # Move data to CPU for metric calculations.
            pred, label = pred.cpu(), label.cpu()

            if 'dice' in self.metrics:
                dice = batch_dice(pred, label)
                self.running_scores['validation-record']['dice'] += dice.item()
                self.running_scores['validation-print']['dice'] += dice.item()

            if 'hausdorff' in self.metrics and iteration > self.hausdorff_delay:
                hausdorff = sitk_batch_hausdorff_distance(pred, label, spacing=self.spacing)
                self.running_scores['print']['hausdorff'] += hausdorff.item()
                self.running_scores['record']['hausdorff'] += hausdorff.item()

            # Print results.
            if self.should_print(self.validation_print_interval, val_batch):
                self.print_validation_results(epoch, val_batch, iteration)
                self.reset_running_scores('validation-print')

        # Check for validation loss improvement.
        loss = self.running_scores['validation-record']['loss'] / self.validation_record_interval
        if loss < self.min_validation_loss:
            # Save model checkpoint.
            info = {
                'training-epoch': epoch,
                'training-iteration': iteration,
                'validation-loss': loss
            }
            checkpoint.save(model, self.run_name, self.optimiser, info=info)
            self.min_validation_loss = loss
            self.num_epochs_since_improvement = 0
        else:
            self.num_epochs_since_improvement += 1
        
        # Record validation results.
        if self.record:
            self.record_validation_results(iteration)
        self.reset_running_scores('validation-record')

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
                if self.record:
                    tag = f"Validation - batch={batch}, view={view}"
                    self.recorder.add_figure(tag, figure, global_step=iteration)

        model.train()
        
    def should_print(self, interval, iteration):
        """
        returns: whether the training or validation score should be printed.
        args:
            interval: the interval between printing.
            iteration: the current training or validation iteration.
        """
        if (iteration + 1) % interval == 0:
            return True
        else:
            return False

    def should_record(self, iteration):
        """
        returns: whether the training score should be recorded.
        args:
            iteration: the current training iteration.
        """
        if not self.record:
            return False
        elif (iteration + 1) % self.train_record_interval == 0:
            return True
        else:
            return False

    def should_validate(self, iteration):
        """
        returns: whether the validation should be performed.
        args:
            iteration: the current training iteration.
        """
        if ((self.validation_interval == 'epoch' and (iteration + 1) % len(self.train_loader) == 0) or
            (self.validation_interval != 'epoch' and (iteration + 1) % self.validation_interval == 0)):
            return True
        else:
            return False

    def print_training_results(self, epoch, batch, iteration):
        """
        effect: logs averaged training results over the last print interval.
        args:
            epoch: the current training epoch.
            batch: the current training batch.
            iteration: the current training iteration.
        """
        # Get average training loss.
        loss = self.running_scores['print']['loss'] / self.train_print_interval
        message = f"[{epoch}, {batch}] Loss: {loss:{PRINT_DP}}"

        # Get additional metrics.
        if 'dice' in self.metrics:
            dice = self.running_scores['print']['dice'] / self.train_print_interval
            message += f", Dice: {dice:{PRINT_DP}}"

        if 'hausdorff' in self.metrics and iteration > self.hausdorff_delay:
            hausdorff = self.running_scores['print']['hausdorff'] / self.train_print_interval
            message += f", Hausdorff: {hausdorff:{PRINT_DP}}"

        self.log_info(message)
        
    def print_validation_results(self, epoch, batch, iteration):
        """
        effect: logs the averaged validation results.
        args:
            epoch: the current training epoch.
            batch: the current validation batch.
            iteration: the current training iteration.
        """
        # Get average validation loss.
        loss = self.running_scores['validation-print']['loss'] / self.validation_print_interval
        message = f"Validation - [{epoch}, {batch}] Loss: {loss:{PRINT_DP}}"

        # Get additional metrics.
        if 'dice' in self.metrics:
            dice = self.running_scores['validation-print']['dice'] / self.validation_print_interval
            message += f", Dice: {dice:{PRINT_DP}}"

        if 'hausdorff' in self.metrics and iteration > self.hausdorff_delay:
            hausdorff = self.running_scores['validation-print']['hausdorff'] / self.validation_print_interval
            message += f", Hausdorff: {hausdorff:{PRINT_DP}}"

        self.log_info(message)

    def record_training_results(self, iteration):
        """
        effect: records averaged training results.
        args:
            iteration: the current training iteration.
        """
        loss = self.running_scores['record']['loss'] / self.train_record_interval
        self.recorder.add_scalar('Loss/train', loss, iteration)
        
        if 'dice' in self.metrics:
            dice = self.running_scores['record']['dice'] / self.train_record_interval
            self.recorder.add_scalar('Dice/train', dice, iteration)

        if 'hausdorff' in self.metrics and iteration > self.hausdorff_delay:
            hausdorff = self.running_scores['record']['hausdorff'] / self.train_record_interval
            self.recorder.add_scalar('Hausdorff/train', hausdorff, iteration)

    def record_validation_results(self, iteration):
        """
        effect: records averaged validation results.
        args:
            iteration: the current training iteration. 
        """
        loss = self.running_scores['validation-record']['loss'] / self.validation_record_interval
        self.recorder.add_scalar('Loss/validation', loss, iteration)

        if 'dice' in self.metrics:
            dice = self.running_scores['validation-record']['dice'] / self.validation_record_interval
            self.recorder.add_scalar('Dice/validation', dice, iteration)

        if 'hausdorff' in self.metrics and iteration > self.hausdorff_delay:
            hausdorff = self.running_scores['record']['hausdorff'] / self.validation_record_interval
            self.recorder.add_scalar('Hausdorff/train', hausdorff, iteration)

    def reset_running_scores(self, key):
        """
        effect: initialises the metrics under the key namespace.
        args:
            key: the metric namespace, e.g. print, record, etc.
        """
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
