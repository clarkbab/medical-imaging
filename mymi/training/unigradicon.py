# Run training for a few epochs on GPU.
# 1. Is it going in the right direction?
# 2. Why are the validation losses nan? Is this just because we haven't passed any validation cases. How do we do this?
# 3. How many epochs does training run for?
# 4. How can we adjust the training code to incorporate a TRE term?

import icon_registration as icon
import os
import torch
import unigradicon as ugi

from mymi.datasets import TrainingDataset
from mymi import logging
from mymi.typing import *

def finetune_unigradicon(
    dataset: DatasetID,
    run_name: str,
    steps: int = int(1e5),     # UGI default.
    ) -> None:
    logging.info(f'Finetuning unigradicon on dataset {dataset} for {steps} steps with run name {run_name}.')

    # Create image loader.
    set = TrainingDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'unigradicon', 'train-samples-spl.pt')
    loaded_images = torch.load(filepath)
    def make_make_batch(images: List[torch.Tensor]):
        def make_batch() -> Tuple[torch.Tensor, torch.Tensor]:  # moving/fixed images.
            draw = np.random.randint(len(images) // 2)
            moving, fixed = images[2 * draw:2 * draw + 2]

            # Process images for network input.
            min_val, max_val = -1000, 1000
            moving, fixed = moving.clamp(min_val, max_val), fixed.clamp(min_val, max_val)
            moving, fixed = (moving - min_val) / (max_val - min_val), (fixed - min_val) / (max_val - min_val)
            moving, fixed = moving.float().cuda(), fixed.float().cuda()
            return moving, fixed
        return make_batch

    # Fine-tune network.
    model = ugi.get_unigradicon()
    model = model.cuda()
    # model = model.cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    model.train()
    icon.train_batchfunction(dataset, run_name, model, optimizer, make_make_batch(loaded_images), steps=steps, unwrapped_net=model)
