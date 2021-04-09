import numpy as np
import os
import torch
from torch.cuda.amp import autocast
from torchio import LabelMap, Subject
from tqdm import tqdm

from mymi import config
from mymi.metrics import dice as dice_metric
from mymi import plotter
from mymi import utils

FILENAME_NUM_DIGITS = 5

class ModelEvaluator:
    def __init__(self, test_loader, device=torch.device('cpu'), metrics=('dice'), mixed_precision=True, pred_spacing=(1, 1, 3), pred_transform=None):
        """
        args:
            test_loader: the loader for the test data.
        kwargs:
            device: the device to train on.
            metrics: the metrics to calculate.
            mixed_precision: whether to use PyTorch mixed precision training.
            pred_spacing: the spacing of the prediction voxels.
            pred_transform: a transform to apply before comparing prediction to the label.
            spacing: the voxel spacing of the data.
        """
        self.device = device
        self.metrics = metrics
        self.mixed_precision = mixed_precision
        self.pred_spacing = pred_spacing
        self.pred_transform = pred_transform
        self.test_loader = test_loader
        self.running_scores = {}
        if 'dice' in metrics:
            self.running_scores['dice'] = 0
            if self.pred_transform is not None:
                self.running_scores['dice-pretransform'] = 0

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

            # Convert prediction into label values.
            pred = pred.argmax(axis=1)

            # Calculate downsampled DSC.
            if 'dice' in self.metrics and self.pred_transform is not None:
                dice = dice_metric(pred, label)
                self.running_scores['dice-pretransform'] += dice.item()

            # Transform prediction before comparing to label.
            if self.pred_transform:
                # Create torchio 'subject'.
                affine = np.array([
                    [self.pred_spacing[0], 0, 0, 0],
                    [0, self.pred_spacing[1], 0, 0],
                    [0, 0, self.pred_spacing[2], 1],
                    [0, 0, 0, 1]
                ])
                pred = LabelMap(tensor=pred, affine=affine)
                subject = Subject(a_segmentation=pred)

                # Transform the subject.
                output = self.pred_transform(subject)

                # Extract results.
                pred = output['a_segmentation'].data

            # Plot the predictions.
            views = ('sagittal', 'coronal', 'axial')
            for view in views:
                centroids = utils.get_batch_centroids(label_raw, view) 
                fig = plotter.plot_batch(input_raw, centroids, figsize=(12, 12), label=label_raw, pred=pred, view=view, return_figure=True)
                filename = f"batch-{batch:0{FILENAME_NUM_DIGITS}}-{view}.png"
                fig.savefig(os.path.join(config.figure_dir, filename))

            # Calculate metrics.
            if 'dice' in self.metrics:
                dice = dice_metric(pred, label_raw)
                self.running_scores['dice'] += dice.item()

        # Print final scores.
        if 'dice' in self.metrics:
            if self.pred_transform is not None:
                mean_dice_pretransform = self.running_scores['dice-pretransform'] / len(self.test_loader)
                print(f"Mean downsampled DSC={mean_dice_pretransform:.6f}")
            mean_dice = self.running_scores['dice'] / len(self.test_loader)
            print(f"Mean DSC={mean_dice:.6f}")
