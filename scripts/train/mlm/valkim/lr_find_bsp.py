from augmed import Pipeline, RandomElastic, RandomCrop, RandomScale, RandomTranslate, RandomGaussianNoise, Standardise, config as amconf
from dicomset import config
from dicomset.utils import logger
from dicomset.typing import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
from typing import *
import wandb

from mymi.loaders.mlm import BSPLoader
from mymi.losses import MSELoss
from mymi.models.architectures.bsp import BSPUNet2DPhase
from mymi.training.utils.lr_find import run_lr_find_bsp
from mymi.utils.interval import interval_matches

def train_bsp(
    dataset: str,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    log_images: bool = False,
    log_images_local: bool = True,
    lr_find: bool = False,
    lr_find_min_lr: float = 1e-7,
    lr_find_max_lr: float = 1,
    lr_find_n_iter: int = 100,
    num_workers: int = 0,
    n_val_images: int = 5,
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    use_logger: bool = True,
    val_image_interval: str = 'epoch:5',
    y_size: int = 768,
) -> None:
    logger.log_method('Training BSP model')
    model_name = model

    # Set seed for reproducible runs.
    torch.manual_seed(random_seed)

    # Create model checkpoint directory.
    ckpt_path = os.path.join(config.dirs.models, project, model_name)
    if os.path.exists(ckpt_path) and not resume and not lr_find:
        logger.info(f"Removing old checkpoint directory {ckpt_path}.")
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)

    # Create pipelines.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"Expected CUDA device, got '{device}'. Ensure a GPU is available."
    amconf.set_dim(2)
    train_pipe = Pipeline([
        RandomElastic(d=(0, 0, -10, 10)),          # Y-deformations only.
        RandomCrop(r=[0, 20, 0, 20, 0, 0, 0, 0]),  # X size changes only.
        RandomScale(s=(1, 1, 0.8, 1.2)),           # Y scaling only.
        RandomTranslate(tp=(0, 0, -0.1, 0.1)),     # Y translations only.
        RandomGaussianNoise(sp=(0, 0.01)),
        Standardise(),
    ], device=device, filter_offgrid=0, fill='border')

    val_pipe = Pipeline([
        Standardise(),
    ], device=device)

    # Create data loaders.
    tl, vl = BSPLoader.build_loaders(
        dataset,
        num_workers=num_workers,
        transform_train=train_pipe,
        transform_val=val_pipe,
    )

    # Create model.
    ckpt_info = {}
    model = BSPUNet2DPhase()
    if resume:
        logger.info(f"Restoring model from checkpoint '{resume_ckpt}'.")
        filepath = os.path.join(ckpt_path, f'{resume_ckpt}.ckpt')
        ckpt_info = torch.load(filepath, map_location=device)
        model.load_state_dict(ckpt_info['model'])
    model.to(device)

    # Define loss function.
    loss_fn = MSELoss()
    logger.info(f"Using loss {loss_fn}.")

    # Create optimiser.
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_init)
    if resume:
        logger.info(f"Restoring optimiser state from checkpoint '{resume_ckpt}'.")
        optimiser.load_state_dict(ckpt_info['optimiser'])

    # --- LR Find ---
    if lr_find:
        run_lr_find_bsp(
            model=model,
            train_loader=tl,
            loss_fn=loss_fn,
            device=device,
            dataset=dataset,
            project=project,
            model_name=model_name,
            min_lr=lr_find_min_lr,
            max_lr=lr_find_max_lr,
            n_iter=lr_find_n_iter,
        )
        return

    # Set up logger.
    run_dir = os.path.join(config.dirs.runs, project, model_name)
    if use_logger:
        run = wandb.init(
            dir=run_dir,
            entity='clarkbab',
            project=project,
            name=model_name,
        )

    # Set up local image save directory.
    image_save_dir = os.path.join(run_dir, 'images')
    if log_images_local:
        os.makedirs(image_save_dir, exist_ok=True)

    start_epoch = ckpt_info['epoch'] + 1 if resume else 0
    step = ckpt_info['step'] + 1 if resume else 0
    val_losses = ckpt_info['val-losses'] if resume else []
    min_val_loss = ckpt_info['min-val-loss'] if resume else np.inf
    val_loss_smoothing = 1
    n_best_ckpts = 1
    best_ckpts = ckpt_info['best-ckpts'] if resume else []

    for e in range(start_epoch, n_epochs):
        # Training loop.
        model.train()
        train_iter = iter(tl)
        for xs, ys in tqdm(train_iter, desc=f'Epoch {e}/{n_epochs} (train)', leave=False):
            xs = xs.to(device)
            ys = ys.to(device)

            ys_pred = model(xs)
            phase = ys[:, 1, :, 1]  # (B, n_frames) in [0, 1]
            phase_sin = torch.sin(2 * torch.pi * phase)
            phase_cos = torch.cos(2 * torch.pi * phase)
            loss = (loss_fn(ys_pred[:, 0], ys[:, 0, :, 1])
                    + loss_fn(ys_pred[:, 1], phase_sin)
                    + loss_fn(ys_pred[:, 2], phase_cos))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if use_logger:
                run.log({
                    'epoch': e,
                    'step': step,
                    'train/loss': loss.item(),
                }, step=step)

            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        with torch.no_grad():
            for i, (xs, ys) in enumerate(tqdm(val_iter, desc=f'Epoch {e}/{n_epochs} (val)', leave=False)):
                xs = xs.to(device)
                ys = ys.to(device)

                ys_pred = model(xs)
                phase = ys[:, 1, :, 1]
                phase_sin = torch.sin(2 * torch.pi * phase)
                phase_cos = torch.cos(2 * torch.pi * phase)
                loss = (loss_fn(ys_pred[:, 0], ys[:, 0, :, 1])
                        + loss_fn(ys_pred[:, 1], phase_sin)
                        + loss_fn(ys_pred[:, 2], phase_cos))
                epoch_val_losses.append(loss.item())

                if use_logger:
                    run.log({
                        'epoch': e,
                        'step': step,
                        'val/loss': loss.item(),
                    }, step=step)

        # Save mean validation loss.
        mean_val_loss = np.mean(epoch_val_losses)
        if use_logger:
            run.log({'val/loss-epoch-mean': mean_val_loss}, step=step)
        val_losses.append(mean_val_loss)

        # Save best model/s.
        if len(val_losses) >= val_loss_smoothing:
            smoothed_val_loss = np.mean(val_losses[-val_loss_smoothing:])
            if use_logger:
                run.log({'val/loss-ma': smoothed_val_loss}, step=step)
            if smoothed_val_loss < min_val_loss:
                min_val_loss = smoothed_val_loss

                # Update 'best_ckpts'.
                ckpt = f'loss={min_val_loss:.6f}_epoch={e}_step={step}'
                best_ckpts.insert(0, ckpt)
                if len(best_ckpts) > n_best_ckpts:
                    old_ckpt = best_ckpts.pop()
                    filepath = os.path.join(ckpt_path, f'{old_ckpt}.ckpt')
                    os.remove(filepath)

                # Save model.
                ckpt_data = {
                    'best-ckpts': best_ckpts,
                    'epoch': e,
                    'min-val-loss': min_val_loss,
                    'model': model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'step': step,
                    'val-losses': val_losses,
                }
                filepath = os.path.join(ckpt_path, f'{ckpt}.ckpt')
                torch.save(ckpt_data, filepath)

        # Save current model.
        ckpt_data = {
            'best-ckpts': best_ckpts,
            'epoch': e,
            'min-val-loss': min_val_loss,
            'model': model.state_dict(),
            'optimiser': optimiser.state_dict(),
            'step': step,
            'val-losses': val_losses,
        }
        filepath = os.path.join(ckpt_path, 'last.ckpt')
        torch.save(ckpt_data, filepath)

if __name__ == '__main__':
    version = '003'
    train_bsp(
        dataset='VALKIM-BSP',
        project='BSP-VALKIM',
        model=f'unet2dphase-{version}',
        n_epochs=100,
        lr_init=1e-5,   # From lr_find results.
        lr_find=True,
        lr_find_min_lr=1e-7,
        lr_find_max_lr=1e-2,
        lr_find_n_iter=1000,
        random_seed=42,
        resume=False,
        use_logger=False,
    )
