import numpy as np
import torch
from torch.cuda.amp import autocast
from torchio import ScalarImage, Subject

from mymi.metrics import dice as dice_metric

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

    def __call__(self, model):
        """
        effect: prints the model evaluation results.
        args:
            model: the model to evaluate.
        """
        # Put model in evaluation mode.
        model.eval()

        for input, label in self.test_loader:
            # Convert input and label.
            input, label = input.float(), label.long()
            input = input.unsqueeze(1)
            input, label = input.to(self.device), label.to(self.device)

            # Perform forward pass.
            with autocast(enabled=self.mixed_precision):
                pred = model(input)
                pred = pred.cpu()

            # Convert prediction into binary values.
            pred = pred.argmax(axis=1)

            # Transform prediction before comparing to label.
            if self.pred_transform:
                # Create torchio 'subject'.
                affine = np.array([
                    [self.pred_spacing[0], 0, 0, 0],
                    [0, self.pred_spacing[1], 0, 0],
                    [0, 0, self.pred_spacing[2], 1],
                    [0, 0, 0, 1]
                ])
                pred = ScalarImage(tensor=pred, affine=affine)
                subject = Subject(one_image=pred)

                # Transform the subject.
                output = self.pred_transform(subject)

                # Extract results.
                pred = output['one_image'].data

            print(pred.shape, label.shape)

            # Calculate metrics.
            if 'dice' in self.metrics:
                dice = dice_metric(pred, label)
                print(dice)


