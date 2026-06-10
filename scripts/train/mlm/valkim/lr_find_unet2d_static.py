from multiprocessing import freeze_support

from dicomset import config
from dicomset.utils import logger
import torch
from typing import *

from mymi.loaders.mlm import StaticLoader
from mymi.losses import DiceLoss
from mymi.models import get_model
from mymi.training.utils.lr_find import run_lr_find_unet2d


def lr_find_unet2d_static(
    dataset: str,
    pat: str,
    project: str,
    model: str,
    n_train_angles: int | None = None,
    n_train_volumes: int | None = None,
    arch: str = 'unet2d:m',
    batch_size: int = 32,
    loss_fn: Literal['dice'] = 'dice',
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
    num_workers: int = 0,
    random_seed: int = 42,
    threshold_labels: bool = True,
    ) -> None:
    logger.log_method('LR find for VALKIM UNet2D static model')
    model_name = model

    torch.manual_seed(random_seed)

    # Create data loaders (val loader unused).
    tl, _ = StaticLoader.build_loaders(
        dataset, pat,
        batch_size=batch_size,
        n_train_angles=n_train_angles,
        n_train_volumes=n_train_volumes,
        num_workers=num_workers,
        threshold_labels=threshold_labels,
    )
    print(f'[lr_find] train_loader_len={len(tl)} batch_size={batch_size} num_workers={num_workers}')

    # Create model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"Expected CUDA device, got '{device}'. Ensure a GPU is available."
    n_output_channels = 2   # GTV + Lung.
    model = get_model(arch, n_output_channels)
    model.to(device)

    # Define loss function.
    if loss_fn == 'dice':
        loss_fn = DiceLoss()
    else:
        raise ValueError(f"Unknown loss function '{loss_fn}'.")

    run_lr_find_unet2d(
        model=model,
        train_loader=tl,
        loss_fn=loss_fn,
        device=device,
        dataset=dataset,
        project=project,
        model_name=model_name,
        min_lr=min_lr,
        max_lr=max_lr,
        n_iter=n_iter,
        n_output_channels=n_output_channels,
    )


if __name__ == '__main__':
    freeze_support()

    pat_id = 'PAT1'
    arch = 'unet2d:m'
    version = '001'
    model = f"{arch.replace(':', '_')}-{pat_id}-{version}"

    lr_find_unet2d_static(
        dataset='VALKIM-PP-STATIC',
        pat=pat_id,
        project='MLM-VALKIM',
        model=model,
        arch=arch,
        batch_size=32,
        loss_fn='dice',
        min_lr=1e-7,
        max_lr=1,
        n_iter=100,
        n_train_angles=361,
        n_train_volumes=100,
        num_workers=4,
        random_seed=42,
        threshold_labels=True,
    )
