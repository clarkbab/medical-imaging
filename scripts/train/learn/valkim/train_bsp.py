from augmed import Pipeline, RandomElastic, RandomCrop, RandomScale, RandomTranslate, RandomGaussianNoise, Standardise, config as amconf
from dicomset import config
from dicomset.utils import logger, plot_slice
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
from mymi.utils.interval import interval_matches

def _get_lr(lr_schedule, epoch: int) -> float:
    """Resolve learning rate for a given epoch from a schedule or constant."""
    if isinstance(lr_schedule, (int, float)):
        return float(lr_schedule)
    current_lr = sorted(lr_schedule, key=lambda x: x[0])[0][1]
    for e, lr in sorted(lr_schedule, key=lambda x: x[0]):
        if e <= epoch:
            current_lr = lr
    return current_lr

def train_bsp(
    dataset: str,
    project: str,
    model: str,
    n_epochs: int,
    lr_schedule: float | List[Tuple[int, float]],
    log_images: bool = False,
    log_images_local: bool = True,
    num_workers: int = 0,
    n_train_images: int = 5,
    n_val_images: int = 5,
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    train_image_interval: str = 'epoch:5',
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
    if os.path.exists(ckpt_path) and not resume:
        logger.info(f"Removing old checkpoint directory {ckpt_path}.")
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)

    # Create pipelines.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amconf.set_dim(2)
    train_pipe = Pipeline([
        RandomElastic(d=(0, 0, -10, 10)),          # Y-deformations only.
        # Can't use resize, as it means we don't have a single point per frame.
        # RandomResize(szp=(0.8, 1.2, 1, 1)),        # X size changes only.
        RandomCrop(r=[0, 20, 0, 20, 0, 0, 0, 0]),  # X size changes only.
        RandomScale(s=(1, 1, 0.8, 1.2)),           # Y scaling only - resize handles this in X.
        RandomTranslate(tp=(0, 0, -0.1, 0.1)),     # Y translations only - x would introduce frames without signal.
        RandomGaussianNoise(sp=(0, 0.01)),         # Adds between 0 and 1% noise to the images.
        Standardise(),  # Only apply to the images.
        # MinMax(only=['points']),  # Only apply to the points.
    ], device=device, filter_offgrid=0, fill='border')
    logger.info(f"Training pipeline:\n{train_pipe}")

    val_pipe = Pipeline([
        Standardise(),
    ], device=device)
    logger.info(f"Validation pipeline:\n{val_pipe}")

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
        model.train()
    else:
        model.to(device)

    # Define loss function (MSE on all channels; phase encoded as sin/cos).
    loss_fn = MSELoss()
    logger.info(f"Using loss {loss_fn}.")

    # Create optimiser.
    optimiser = torch.optim.AdamW(model.parameters(), lr=_get_lr(lr_schedule, 0))
    if resume:
        logger.info(f"Restoring optimiser state from checkpoint '{resume_ckpt}'.")
        optimiser.load_state_dict(ckpt_info['optimiser'])

    # Set up logger.
    run_dir = os.path.join(config.dirs.runs, project, model_name)
    if use_logger:
        run = wandb.init(
            dir=run_dir,
            entity='clarkbab',
            project=project,
            name=model_name,
        )

    # Set up local image save directories.
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
        # Update learning rate.
        epoch_lr = _get_lr(lr_schedule, e)
        for pg in optimiser.param_groups:
            pg['lr'] = epoch_lr
        if use_logger:
            run.log({'train/lr': epoch_lr}, step=step)

        # Training loop.
        model.train()
        train_iter = iter(tl)
        for i, (xs, ys) in enumerate(tqdm(train_iter, desc=f'Epoch {e}/{n_epochs} (train)', leave=False)):
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

            # Log train images.
            if log_images and interval_matches(step, train_image_interval, len(tl), step_match_length=n_train_images):
                x_np = xs[0, 0].cpu().numpy()  # (n_frames, y_size)
                n_frames = x_np.shape[0]
                xs_pts = np.arange(n_frames)

                gt_pts_list = []
                pred_pts_list = []

                amp_gt = ys[0, 0, :, 1].cpu().numpy()
                amp_pred = ys_pred[0, 0].detach().cpu().numpy()
                gt_pts_list.append(np.stack([xs_pts, amp_gt], axis=1))
                pred_pts_list.append(np.stack([xs_pts, amp_pred], axis=1))

                phase_gt = ys[0, 1, :, 1].cpu().numpy()
                sin_pred = ys_pred[0, 1].detach().cpu().numpy()
                cos_pred = ys_pred[0, 2].detach().cpu().numpy()
                phase_pred = (np.arctan2(sin_pred, cos_pred) / (2 * np.pi)) % 1.0
                gt_pts_list.append(np.stack([xs_pts, phase_gt], axis=1))
                pred_pts_list.append(np.stack([xs_pts, phase_pred], axis=1))

                gt_pts_list = [normalise_points(p, min=0, max=y_size - 1) for p in gt_pts_list]
                pred_pts_list = [normalise_points(p, min=0, max=y_size - 1) for p in pred_pts_list]

                fig, axs = plt.subplots(2, 1, figsize=(16, 8))
                plot_slice(x_np, ax=axs[0], points=gt_pts_list, title='GT', aspect=0.1, orientation='LI')
                plot_slice(x_np, ax=axs[1], points=pred_pts_list, title='Pred', aspect=0.1, orientation='LI')
                fig.tight_layout()
                filename = f'epoch-{e:03d}-step-{step:06d}-train-{i:03d}.png'
                if log_images_local:
                    fig.savefig(os.path.join(train_image_save_dir, filename), dpi=150, bbox_inches='tight')
                else:
                    run.log({'train/image': wandb.Image(fig)}, step=step)
                plt.close(fig)

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
                phase = ys[:, 1, :, 1]  # (B, n_frames) in [0, 1]
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

                # Log images.
                if log_images and interval_matches(step, val_image_interval, len(val_iter), step_match_length=n_val_images):
                    x_np = xs[0, 0].cpu().numpy()  # (n_frames, y_size)
                    n_frames = x_np.shape[0]
                    xs_pts = np.arange(n_frames)

                    # Build (n_frames, 2) points per signal, normalised to image y-size.
                    gt_pts_list = []
                    pred_pts_list = []

                    # Amplitude (channel 0): direct.
                    amp_gt = ys[0, 0, :, 1].cpu().numpy()
                    amp_pred = ys_pred[0, 0].cpu().numpy()
                    gt_pts_list.append(np.stack([xs_pts, amp_gt], axis=1))
                    pred_pts_list.append(np.stack([xs_pts, amp_pred], axis=1))

                    # Phase (channels 1, 2): decode sin/cos → [0, 1].
                    phase_gt = ys[0, 1, :, 1].cpu().numpy()
                    sin_pred = ys_pred[0, 1].cpu().numpy()
                    cos_pred = ys_pred[0, 2].cpu().numpy()
                    phase_pred = (np.arctan2(sin_pred, cos_pred) / (2 * np.pi)) % 1.0
                    gt_pts_list.append(np.stack([xs_pts, phase_gt], axis=1))
                    pred_pts_list.append(np.stack([xs_pts, phase_pred], axis=1))

                    # Normalise y-coords to image pixel space.
                    gt_pts_list = [normalise_points(p, min=0, max=y_size - 1) for p in gt_pts_list]
                    pred_pts_list = [normalise_points(p, min=0, max=y_size - 1) for p in pred_pts_list]

                    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
                    plot_slice(x_np, ax=axs[0], points=gt_pts_list, title='GT', aspect=0.1, orientation='LI')
                    plot_slice(x_np, ax=axs[1], points=pred_pts_list, title='Pred', aspect=0.1, orientation='LI')

                    fig.tight_layout()
                    filename = f'epoch-{e:03d}-step-{step:06d}-val-{i:03d}.png'
                    if log_images_local:
                        fig.savefig(os.path.join(val_image_save_dir, filename), dpi=150, bbox_inches='tight')
                    else:
                        run.log({'val/image': wandb.Image(fig)}, step=step)
                    plt.close(fig)

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

def normalise_points(
    points: Points2D | BatchPoints2D,
    min: float = 0,
    max: float = 1,
    ) -> Points2D | BatchPoints2D:
    return_batch = True
    if points.ndim == 2:
        points = points[np.newaxis]
        return_batch = False
    
    norm_points = []
    for p in points:
        # y-axis only.
        y_min, y_max = p[:, 1].min(), p[:, 1].max()
        p[:, 1] = (max - min) * (p[:, 1] - y_min) / (y_max - y_min) + min
        norm_points.append(p)
    norm_points = np.stack(norm_points, axis=0)

    if not return_batch:
        norm_points = norm_points[0]

    return norm_points

if __name__ == '__main__':
    version = '006'
    train_bsp(
        dataset='VALKIM-BSP',
        project='BSP-VALKIM',
        model=f'unet2dphase-{version}',
        n_epochs=500,
        lr_schedule=[
            (0, 1e-4),
            (200, 1e-5),
        ],
        log_images=True,
        log_images_local=True,
        random_seed=42,
        resume=False,
        use_logger=True,
    )
