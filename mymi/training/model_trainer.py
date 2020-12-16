import logging
import torch

from mymi.metrics import dice as dice_metric

class ModelTrainer:
    def __init__(self, train_loader, validation_loader, optimiser, loss_fn, num_epochs=1, metrics=('dice'),
        device=torch.device('cpu'), print_interval='epoch', record_interval='epoch', print_format='.10f'):
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs
        self.metrics = metrics
        self.device = device
        self.print_interval = print_interval
        self.record_interval = record_interval
        self.print_format = print_format

    def __call__(self, model):
        """
        effect: updates the model parameters.
        model: the model to train.
        """
        for epoch in range(self.num_epochs):
            # Keep tally for printing and recording averages.
            running_scores = {
                'print': {
                    'loss': 0
                },
                'record': {
                    'loss': 0
                }
            }
            if 'dice' in self.metrics: 
                running_scores['print']['dice'] = 0
                running_scores['record']['dice'] = 0

            for batch, (input, mask) in enumerate(self.train_loader):
                input, mask = input.float(), mask.long()
                input = input.unsqueeze(1)
                input, mask = input.to(self.device), mask.to(self.device)

                # Perform forward/backward pass.
                self.optimiser.zero_grad()
                pred = model(input)
                loss = self.loss_fn(pred, mask)
                loss.backward()
                self.optimiser.step()
                running_scores['print']['loss'] += loss.item()
                running_scores['record']['loss'] += loss.item()

                # Calculate other metrics.
                if 'dice' in self.metrics:
                    dice = dice_metric(pred, mask)
                    running_scores['print']['dice'] += dice.item()
                    running_scores['record']['dice'] += dice.item()

                # Record info.
                if self.record_interval != 'epoch' and batch % self.record_interval == 0:
                    row_data = {
                        'time': time.time(),
                        'epoch': epoch,
                        'batch': batch,
                        'loss': running_scores['record']['loss'] / self.record_interval
                    }
                    running_scores['record']['loss'] = 0
                    
                    if 'dice' in self.metrics:
                        row_data['dice'] = running_scores['record']['dice'] / self.record_interval
                        running_scores['record']['dice'] = 0

                    data_df = data_df.append(row_data, ignore_index=True)

                # Print info.
                if self.print_interval != 'epoch' and batch % self.print_interval == 0:
                    self.log_results(epoch, batch, running_scores)
                    running_scores['print']['loss'] = 0
                    if 'dice' in self.metrics: running_scores['print']['dice'] = 0

            # Record info.

            # Print info.
            if self.print_interval == 'epoch':
                self.log_results(epoch, batch, running_scores)
                running_scores['print']['loss'] = 0
                if 'dice' in self.metrics: running_scores['print']['dice'] = 0

    def log_results(self, epoch, batch, running_scores):
        """
        effect: logs an update to STDOUT.
        """
        logging.info(f"[{epoch}, {batch}] loss: {running_scores['print']['loss'] / self.print_interval:{self.print_format}}")

        if 'dice' in self.metrics:
            logging.info(f"Dice: {running_scores['print']['dice'] / self.print_interval:{self.print_format}}")
