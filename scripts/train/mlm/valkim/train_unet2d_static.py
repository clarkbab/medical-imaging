from multiprocessing import freeze_support

from dicomset import config
from dicomset.utils import logger
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

from mymi.loaders.mlm import StaticLoader
from mymi.losses import DiceLoss
from mymi.models import get_model, load_model
from mymi.training.utils.lr_find import run_lr_find_unet2d
from mymi.utils.interval import interval_matches

def train_unet2d_static(
    dataset: str,
    pat: str,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    n_train_angles: int | None = None,
    n_train_volumes: int | None = None,
    n_val_angles: int | None = None,
    n_val_volumes: int | None = None,
    arch: str = 'unet2d:m',
    batch_size: int = 32,
    log_images: bool = False,
    log_images_local: bool = True,
    loss_fn: Literal['dice'] = 'dice',
    lr_find: bool = False,
    lr_find_min_lr: float = 1e-7,
    lr_find_max_lr: float = 1,
    lr_find_n_iter: int = 100,
    n_train_images: int = 5,
    n_val_images: int = 5,
    num_workers: int = 0,
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    threshold_labels: bool = True,
    train_image_interval: str = 'epoch:5',
    use_logging: bool = True,
    val_image_interval: str = 'epoch:5',
    ) -> None:
    logger.log_method('Training VALKIM segmentation model')
    model_name = model

    torch.manual_seed(random_seed)

    ckpt_path = os.path.join(config.dirs.models, project, model_name)
    if os.path.exists(ckpt_path) and not resume and not lr_find:
        logger.info(f"Removing old checkpoint directory {ckpt_path}.")
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)

    # Create data loaders.
    tl, vl = StaticLoader.build_loaders(
        dataset, pat,
        batch_size=batch_size,
        n_train_angles=n_train_angles,
        n_train_volumes=n_train_volumes,
        n_val_angles=n_val_angles,
        n_val_volumes=n_val_volumes,
        num_workers=num_workers,
        threshold_labels=threshold_labels,
    )
    print(f'[train] train_loader_len={len(tl)} val_loader_len={len(vl)} batch_size={batch_size} num_workers={num_workers}')

    # Create model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"Expected CUDA device, got '{device}'. Ensure a GPU is available."
    n_output_channels = 2   # GTV plus background.
    model = get_model(arch, n_output_channels)
    ckpt_info = {}
    if resume:
        model, _, ckpt_info = load_model(model, project, model_name, resume_ckpt, device=device, state='train')
    else:
        model.to(device)

    # Define loss function.
    if loss_fn == 'dice':
        loss_fn = DiceLoss()
    else:
        raise ValueError(f"Unknown loss function '{loss_fn}'.")
    logger.info(f"Using loss {loss_fn}.")

    # --- LR Find ---
    if lr_find:
        run_lr_find_unet2d(
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

    # Create optimiser.
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_init)
    if resume:
        logger.info(f"Restoring optimiser state from checkpoint {resume_ckpt}.")
        optimiser.load_state_dict(ckpt_info['optimiser'])

    # Set up logging.
    run_dir = os.path.join(config.dirs.runs, project, model_name)
    if use_logging:
        run = wandb.init(
            dir=run_dir,
            entity='clarkbab',
            project=project,
            name=model_name,
        )

    train_image_save_dir = os.path.join(run_dir, 'images', 'train')
    val_image_save_dir = os.path.join(run_dir, 'images', 'val')
    if log_images_local:
        os.makedirs(train_image_save_dir, exist_ok=True)
        os.makedirs(val_image_save_dir, exist_ok=True)

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
        for i, (xs, ys, angles) in enumerate(tqdm(train_iter, desc=f'Epoch {e}/{n_epochs} (train)', leave=False, total=len(tl))):
            if i == 0:
                print(f'[train] first batch shapes xs={tuple(xs.shape)} ys={tuple(ys.shape)}')
            xs = xs.to(device)
            ys = ys.to(device)
            ys = ys[:, :n_output_channels]

            ys_pred = model(xs)
            loss = loss_fn(ys_pred, ys)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if use_logging:
                run.log({
                    'epoch': e,
                    'step': step,
                    'train/loss': loss.item(),
                }, step=step)

            # Log train images.
            if log_images and interval_matches(step, train_image_interval, len(tl), step_match_length=n_train_images):
                x_slice = xs[0, 0].cpu().numpy()
                y_slice = ys[0, 1].cpu().numpy()
                y_pred_slice = (ys_pred[0, 1].detach().cpu().numpy() > 0.5).astype(np.float32)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(x_slice.T, cmap='gray', origin='lower')
                axes[0].set_title('Input')
                axes[0].axis('off')
                axes[1].imshow(y_slice.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                axes[1].set_title('Ground Truth (GTV)')
                axes[1].axis('off')
                axes[2].imshow(y_pred_slice.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                axes[2].set_title('Prediction (GTV)')
                axes[2].axis('off')
                fig.tight_layout()

                filename = f'epoch-{e:04d}-step-{step:06d}-train-{i:03d}-GTV.png'
                if log_images_local:
                    fig.savefig(os.path.join(train_image_save_dir, filename), dpi=150, bbox_inches='tight')
                else:
                    run.log({'train/image-GTV': wandb.Image(fig)}, step=step)
                plt.close(fig)

            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        if e == start_epoch:
            print(f'[train] validation_loader_len={len(vl)}')
        for i, (xs, ys, angles) in enumerate(tqdm(val_iter, desc=f'Epoch {e}/{n_epochs} (val)', leave=False, total=len(vl))):
            xs = xs.to(device)
            ys = ys.to(device)
            ys = ys[:, :n_output_channels]

            with torch.no_grad():
                ys_pred = model(xs)
                loss = loss_fn(ys_pred, ys)

            epoch_val_losses += [loss.item()]

            if use_logging:
                run.log({
                    'epoch': e,
                    'step': step,
                    'val/loss': loss.item(),
                }, step=step)

            # Log val images.
            if log_images and interval_matches(step, val_image_interval, len(vl)) and i < n_val_images:
                x_slice = xs[0, 0].cpu().numpy()
                y_slice = ys[0, 1].cpu().numpy()
                y_pred_slice = (ys_pred[0, 1].detach().cpu().numpy() > 0.5).astype(np.float32)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(x_slice.T, cmap='gray', origin='lower')
                axes[0].set_title('Input')
                axes[0].axis('off')
                axes[1].imshow(y_slice.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                axes[1].set_title('Ground Truth (GTV)')
                axes[1].axis('off')
                axes[2].imshow(y_pred_slice.T, cmap='gray', vmin=0, vmax=1, origin='lower')
                axes[2].set_title('Prediction (GTV)')
                axes[2].axis('off')
                fig.tight_layout()

                filename = f'epoch-{e:04d}-step-{step:06d}-val-{i:03d}-GTV.png'
                if log_images_local:
                    fig.savefig(os.path.join(val_image_save_dir, filename), dpi=150, bbox_inches='tight')
                else:
                    run.log({'val/image-GTV': wandb.Image(fig)}, step=step)
                plt.close(fig)

        # Save mean validation loss.
        mean_val_loss = np.mean(epoch_val_losses)
        if use_logging:
            run.log({'val/loss-epoch-mean': mean_val_loss}, step=step)
        val_losses.append(mean_val_loss)

        # Save best model/s.
        if len(val_losses) >= val_loss_smoothing:
            smoothed_val_loss = np.mean(val_losses[-val_loss_smoothing:])
            if use_logging:
                run.log({'val/loss-ma': smoothed_val_loss}, step=step)
            if smoothed_val_loss < min_val_loss:
                min_val_loss = smoothed_val_loss

                ckpt = f'loss={min_val_loss:.6f}_epoch={e}_step={step}'
                best_ckpts.insert(0, ckpt)
                if len(best_ckpts) > n_best_ckpts:
                    old_ckpt = best_ckpts.pop()
                    filepath = os.path.join(ckpt_path, f'{old_ckpt}.ckpt')
                    os.remove(filepath)

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
                filepath = os.path.join(ckpt_path, 'best.ckpt')
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
        filepath = os.path.join(ckpt_path, f'epoch={e:04d}.ckpt')
        torch.save(ckpt_data, filepath)

if __name__ == '__main__':
    freeze_support()

    pat_id = 'PAT1'
    arch = 'unet2d:m'
    version = '002'
    model = f"{arch.replace(':', '_')}-{pat_id}-{version}"

    train_unet2d_static(
        dataset='VALKIM-PP-STATIC',
        pat=pat_id,
        project='MLM-VALKIM',
        model=model,
        n_epochs=100,
        # lr_init=1e-3,
        lr_init=5e-4,   # From the LR find.
        arch=arch,
        batch_size=32,
        log_images=True,
        log_images_local=True,
        loss_fn='dice',
        lr_find=False,
        n_train_angles=361,
        n_train_volumes=100,
        n_val_angles=100,
        n_val_volumes=10,
        num_workers=4,
        random_seed=42,
        resume=False,
        threshold_labels=True,
        train_image_interval='step:200',
        val_image_interval='step:epoch_start',
        use_logging=True,
    )
