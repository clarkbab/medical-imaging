import numpy as np
import os
import torch
from torch.cuda.amp import autocast
from torchio import LabelMap, Subject
from tqdm import tqdm

from mymi import config
from mymi.metrics import batch_dice
from mymi import plotter
from mymi import utils

class ModelEvaluator:
    def __init__(self, run_name, test_loader, device=torch.device('cpu'), metrics=('dice'), mixed_precision=True, output_spacing=None, output_transform=None, save_data=False):
        """
        args:
            run_name: the name of the run.
            test_loader: the loader for the test data.
        kwargs:
            device: the device to train on.
            metrics: the metrics to calculate.
            mixed_precision: whether to use PyTorch mixed precision training.
            output_spacing: the voxel spacing of the input data.
            output_transform: the transform to apply before comparing prediction to label.
            save_data: whether to save predictions, figures, etc. to disk.
        """
        self.device = device
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.output_spacing = output_spacing
        self.output_transform = output_transform
        if output_transform:
            assert output_spacing, 'Output spacing must be specified if output transform applied.'
        self.run_name = run_name
        self.test_loader = test_loader

        # Initialise running scores.
        self.running_scores = {}
        if 'dice' in metrics:
            self.running_scores['dice'] = 0
            if self.output_transform is not None:
                self.running_scores['output-dice'] = 0

    def __call__(self, model):
        """
        effect: prints the model evaluation results.
        args:
            model: the model to evaluate.
        """
        # Put model in evaluation mode.
        model.eval()

        for batch, (input, label, input_raw, label_raw) in enumerate(tqdm(self.test_loader)):
            # Convert input and label.
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            with autocast(enabled=self.mixed_precision):
                pred = model(input)

            # Move data back to cpu for calculations.
            pred, label = pred.cpu(), label.cpu()

            # Convert prediction into binary values.
            pred = pred.argmax(axis=1)

            # Save output predictions and labels.
            folder = 'output' if self.output_transform else 'raw'
            filepath = os.path.join(config.directories.evaluation, self.run_name, 'predictions', folder, f"batch-{batch}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.save(filepath, pred.numpy().astype(np.bool))
            filepath = os.path.join(config.directories.evaluation, self.run_name, 'labels', folder, f"batch-{batch}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            np.save(filepath, label.numpy().astype(np.bool))

            # Calculate output DSC.
            if 'dice' in self.metrics and self.output_transform:
                dice = batch_dice(pred, label)
                self.running_scores['output-dice'] += dice.item()

            # Transform prediction before comparing to label.
            if self.output_transform:
                # Create torchio 'subject'.
                affine = np.array([
                    [self.output_spacing[0], 0, 0, 0],
                    [0, self.output_spacing[1], 0, 0],
                    [0, 0, self.output_spacing[2], 1],
                    [0, 0, 0, 1]
                ])
                pred = LabelMap(tensor=pred, affine=affine)
                subject = Subject(a_segmentation=pred)

                # Transform the subject.
                output = self.output_transform(subject)

                # Extract results.
                pred = output['a_segmentation'].data

                # Save transformed predictions and labels.
                filepath = os.path.join(config.directories.evaluation, self.run_name, 'predictions', 'raw', f"batch-{batch}")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.save(filepath, pred.numpy().astype(bool))
                filepath = os.path.join(config.directories.evaluation, self.run_name, 'labels', 'raw', f"batch-{batch}")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                np.save(filepath, label_raw.numpy().astype(bool))

            # Plot the predictions.
            views = ('sagittal', 'coronal', 'axial')
            for view in views:
                # Find central slices.
                centroids = utils.get_batch_centroids(label_raw, view) 

                # Create and save figures.
                fig = plotter.plot_batch(input_raw, centroids, figsize=(12, 12), label=label_raw, pred=pred, view=view, return_figure=True)
                filepath = os.path.join(config.directories.evaluation, self.run_name, 'figures', f"batch-{batch:0{config.formatting.sample_digits}}-{view}.png")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                fig.savefig(filepath)

            # Calculate metrics.
            if 'dice' in self.metrics:
                dice = batch_dice(pred, label_raw)
                self.running_scores['dice'] += dice.item()

        # Print final scores.
        if 'dice' in self.metrics:
            if self.output_transform is not None:
                mean_output_dice = self.running_scores['output-dice'] / len(self.test_loader)
                print(f"Mean output DSC={mean_output_dice:{config.formatting.metrics}}")
            mean_dice = self.running_scores['dice'] / len(self.test_loader)
            print(f"Mean DSC={mean_dice:{config.formatting.metrics}}")
