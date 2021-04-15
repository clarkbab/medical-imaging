import os
import torch

from mymi import config

class Checkpoint:
    @classmethod
    def load(cls, model_name, checkpoint_name='checkpoint'):
        """
        returns: the saved checkpoint data.
        args:
            name: the name of the model to load.
        kwargs:
            checkpoint_name: the name of the checkpoint.
        """
        # Load data.
        filepath = os.path.join(config.directories.checkpoints, model_name, f"{checkpoint_name}.pt")
        f = open(filepath, 'rb')
        data = torch.load(f)

        return data

    @classmethod
    def save(cls, model, model_name, optimiser, checkpoint_name='checkpoint', info=None):
        """
        effect: saves a copy of the model and optimiser state.
        args:
            model: the model to save.
            model_name: the name of the model.
            optimiser: the optimiser used for training.
        kwargs:
            checkpoint_name: the name of the checkpoint.
            info: additional info to save.
        """
        # Create data dict.
        data = {
            'model_state': model.state_dict(),
            'optimiser_state': optimiser.state_dict()
        }

        # Save model and optimiser data.
        filepath = os.path.join(config.directories.checkpoints, model_name, f"{checkpoint_name}.pt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(data, filepath)

        # Save additional info.
        if info:
            filepath = os.path.join(config.directories.checkpoints, model_name, f"{checkpoint_name}.csv")
            with open(filepath, 'w') as f:
                for key in info.keys():
                    f.write(f"{key},{info[key]}")
