import os
from torch.utils.tensorboard import SummaryWriter

from mymi import config

class Recorder:
    def __init__(self, name):
        """
        args:
            name: the name of the training run.
        """
        self.recorder = SummaryWriter(os.path.join(config.directories.tensorboard, name))

    def record_figure(self, tag, figure, iteration):
        """
        effect: records a figure.
        args:
            tag: the tag to identify the figure.
            figure: the 'matplotlib.figure.Figure' to show.
            iteration: the iteration - to track changes in the tagged figure over time.
        """
        self.recorder.add_figure(tag, figure, global_step=iteration)

    def record_metric(self, tag, value, iteration):
        """
        effect: records a metric.
        args:
            tag: the metric tag.
            value: the metric value.
            iteration: the iteration - to track changes to the metric over time.
        """
        self.recorder.add_scalar(tag, value, iteration)

    def record_model_graph(self, model, input):
        """
        effect: records the model graph and layer dimensions.
        args:
            model: the model to record.
            input: a sample input.
        """
        self.recorder.add_graph(model, input)

    def record_training_params(self, params):
        """
        effect: records the parameters used for training.
        args:
            params: the parameters used for training.
        """
        self.recorder.add_hparams(params, {}, run_name='hparams')
    