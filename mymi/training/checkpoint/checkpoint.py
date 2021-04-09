import os
import torch

from mymi import config

class Checkpoint:
    @classmethod
    def load(cls, name):
        """
        returns: the loaded model and optimiser data.
        args:
            name: the name of the model to load.
        """
        # Load data.
        filepath = os.path.join(config.checkpoint_dir, name, 'best.pt')
        f = open(filepath, 'rb')
        data = torch.load(f)

        return data

    @classmethod
    def save(cls, model, optimiser, name):
        """
        effect: saves a copy of the model and optimiser state.
        args:
            model: the model to save.
            optimiser: the optimiser used for training.
            name: the model name.
        """
        # Create data dict.
        data = {
            'model_state': model.state_dict(),
            'optimiser_state': optimiser.state_dict()
        }

        # Save data.
        filepath = os.path.join(config.checkpoint_dir, name, 'best.pt')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(data, filepath)
