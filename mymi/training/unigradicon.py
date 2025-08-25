import icon_registration as icon
import os
import torch
import unigradicon as ugi

from mymi.constants import *
from mymi.datasets import TrainingDataset
from mymi.loaders.data_augmentation import RandomAffine, RandomFlip
from mymi import logging
from mymi.transforms import resample
from mymi.typing import *

def finetune_unigradicon(
    dataset: DatasetID,
    run_name: str,
    lr: float = 0.00005,
    steps: int = int(1e5),     # UGI default.
    ) -> None:
    logging.info(f'Finetuning unigradicon on dataset {dataset} for {steps} steps with run name {run_name}.')
    fill_val = -2000

    # Define data augmentation.
    data_augs = [
        RandomAffine(),
        RandomFlip(),
    ]

    # Create image loader.
    set = TrainingDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'unigradicon', 'train-samples.pt')
    loaded_images, loaded_spacings = torch.load(filepath, weights_only=False)
    assert len(loaded_images) == len(loaded_spacings), "Loaded images and spacings must have the same length."
    def make_make_batch(
        images: List[torch.Tensor],
        spacings: List[Spacing3D]) -> Callable[[], Tuple[torch.Tensor, torch.Tensor]]:
        def make_batch() -> Tuple[torch.Tensor, torch.Tensor]:  # moving/fixed images.
            draw = np.random.randint(len(images) // 2)
            moving, fixed = images[2 * draw:2 * draw + 2]
            moving_spacing, fixed_spacing = spacings[2 * draw:2 * draw + 2]
            assert np.array_equal(moving_spacing, fixed_spacing), "Moving and fixed spacings must be equal."
            spacing = fixed_spacing
            offset = (0, 0, 0)

            # Create composite transforms.
            if len(data_augs) > 0:
                t_fs = []
                t_bs = []
                for d in data_augs:
                    transform_f, transform_b, _ = d.get_concrete_transform(size=moving.shape[2:], spacing=spacing, offset=offset)
                    t_bs.insert(0, transform_b)
                    t_fs.append(transform_f)
                transform_f = sitk.CompositeTransform(t_fs)
                transform_b = sitk.CompositeTransform(t_bs)
            else:
                transform_f = None
                transform_b = None

            if transform_b is not None:
                # Transform moving/fixed images.
                moving = resample(moving, fill=fill_val, spacing=spacing, offset=offset, transform=transform_b)
                fixed = resample(fixed, fill=fill_val, spacing=spacing, offset=offset, transform=transform_b)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    icon.train_batchfunction(dataset, run_name, model, optimizer, make_make_batch(loaded_images, loaded_spacings), steps=steps, unwrapped_net=model)
