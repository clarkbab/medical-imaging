import os
import torch
from typing import *

from mymi import config

class Checkpoint:
    @classmethod
    def load(
        cls,
        model_name: str,
        run_name: str,
        checkpoint_name: str = 'checkpoint',
        device: torch.device = torch.device('cpu')) -> Tuple[dict, dict]:
        """
        returns: the saved checkpoint model and optimiser states.
        args:
            model_name: the name of the model to load.
            run_name: the name of the training run.
        kwargs:
            checkpoint_name: the name of the checkpoint.
        """
        # Load data.
        filepath = os.path.join(config.directories.checkpoints, model_name, run_name, f"{checkpoint_name}.pt")
        f = open(filepath, 'rb')
        data = torch.load(f, map_location=device)

        # Pull out keys.
        model_state = data['model_state']
        optimiser_state = data['optimiser_state']

        return model_state, optimiser_state

    @classmethod
    def save(
        cls,
        model: torch.nn.Module,
        model_name: str,
        optimiser: torch.optim,
        run_name: str,
        checkpoint_name: str = 'checkpoint',
        info: dict = None) -> None:
        """
        effect: saves a copy of the model and optimiser state.
        args:
            model: the model to save.
            model_name: the name of the model.
            optimiser: the optimiser used for training.
            run_name: the name of the training run.
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
        dirpath = os.path.join(config.directories.checkpoints, model_name, run_name)
        filepath = os.path.join(dirpath, f"{checkpoint_name}.pt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(data, filepath)

        # Save additional info.
        if info:
            filepath = os.path.join(dirpath, f"{checkpoint_name}.csv")
            with open(filepath, 'w') as f:
                for key in info.keys():
                    f.write(f"{key},{info[key]}")
