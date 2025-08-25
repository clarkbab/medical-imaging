import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
from typing import *
import wandb

from mymi import config
from mymi.loaders.data_augmentation import RandomAffine
from mymi.loaders import HoldoutLoader
from mymi import logging
from mymi.losses import MSELoss, NCCLoss, SpatialSmoothingLoss
from mymi.losses.voxelmorph import Grad
from mymi.models import load_model
from mymi.models.architectures import RegMod
from mymi.utils import *

def train_registration(
    dataset: str,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    loss_fn: Literal['mse', 'ncc'] = 'mse',
    loss_lambda: float = 0.02,
    n_val: Optional[int] = None,
    n_workers: int = 1,
    pad_fill: Optional[float] = -1024,
    pad_threshold: Optional[float] = -1024,
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    use_logging: bool = True,
    val_image_interval: str = 'epoch:5') -> None:
    logging.arg_log('Training registration model', ('dataset', 'project', 'model'), (dataset, project, model))
    model_name = model  # Use model for actual model.

    # Set seed for reproducible runs.
    torch.manual_seed(random_seed)

    ckpt_path = os.path.join(config.directories.models, project, model_name)
    if os.path.exists(ckpt_path) and not resume:
        # Clean up old run files.
        logging.info(f"Removing old checkpoint directory {ckpt_path}.")
        shutil.rmtree(ckpt_path)
    os.makedirs(ckpt_path, exist_ok=True)

    # Create data loaders.
    data_aug = RandomAffine()
    okwargs = dict(
        batch_size=1,
        n_val=n_val,
        n_workers=n_workers,
        pad_fill=pad_fill,
        pad_threshold=pad_threshold,
        transform_train=data_aug,
    )
    tl, vl, _, norm_params = HoldoutLoader.build_loaders(dataset, **okwargs)

    # Create model.
    # Issue with setting pad_value=min, is that DVFs can point offscreen to get air-like
    # intensity values. Using border padding to stop this behaviour. With border padding,
    # a DVF can always achieve the same similarity score as an offscreen (reflected) voxel
    # by choosing an onscreen voxel that is closer (although we don't directly penalise
    # DVF magnitudes). This is only not the case for border voxels, which could be equally
    # incentivised to select an offscreen voxel as an onscreen one...
    device = torch.device('cuda')
    module = RegMod
    if resume:
        model, _, ckpt_info = load_model(module, project, model_name, resume_ckpt, device=device, state='train')
        if ckpt_info['norm-params'] != norm_params:
            raise ValueError(f"Got different norm params for loader {norm_params} and model {ckpt_info['norm-params']}.") 
    else:
        model = module()
        model.to(device)

    # Define loss function.
    if loss_fn == 'mse':
        loss_sim = MSELoss()
    elif loss_fn == 'ncc':
        loss_sim = NCCLoss()
    else:
        raise ValueError(f"Unknown loss function '{loss_fn}'.")
    logging.info(f"Using sim loss {loss_sim}.")
    # loss_smooth = SpatialSmoothingLoss()
    loss_smooth = Grad(penalty='l2')
    logging.info(f"Using smooth loss {loss_smooth} with lambda={loss_lambda}.")

    # Create optimiser.
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr_init) 
    if resume:
        logging.info(f"Restoring optimiser state from checkpoint {resume_ckpt}.")
        optimiser.load_state_dict(ckpt_info['optimiser'])

    # Set up logging.
    if use_logging:
        run = wandb.init(
            dir=config.directories.reports,
            entity="clarkbab",
            project=project,
            name=model_name,
        )

    # Checkpoints are saved at the end of the epoch/step, so need to increment values by 1.
    start_epoch = ckpt_info['epoch'] + 1 if resume else 0
    step = ckpt_info['step'] + 1 if resume else 0
    val_losses = ckpt_info['val-losses'] if resume else [] # Calculating moving average for checkpointing - less noisy.
    min_val_loss = ckpt_info['min-val-loss'] if resume else np.inf
    val_loss_smoothing = 1
    n_best_ckpts = 1
    best_ckpts = ckpt_info['best-ckpts'] if resume else []
    for epoch in range(start_epoch, n_epochs):
        # Training loop.
        model.train()
        train_iter = iter(tl)
        for desc, input, *labels in tqdm(train_iter, desc=f'Epoch {epoch}/{n_epochs} (train)', leave=False):
            input = input.to(device)
            labels = [l.to(device) for l in labels]

            # Perform training update.
            y_moved, y_dvf = model(input)
            x_fixed = input[:, 0].unsqueeze(1)  # Single-channel image should still have channel dimension.
            sim_loss = loss_sim(y_moved, x_fixed)
            smooth_loss = loss_smooth(y_dvf) 
            scaled_smooth_loss = loss_lambda * smooth_loss
            loss = sim_loss + scaled_smooth_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Record metrics.
            if use_logging:
                run.log({
                    'epoch': epoch,
                    'step': step,
                    'train/loss': loss.item(),
                    'train/loss-sim': sim_loss.item(),
                    'train/loss-smooth': scaled_smooth_loss.item(),
                    'train/loss-smooth-raw': smooth_loss.item(),
                }, step=step)

            # Increment training step.
            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        for desc, input, labels in tqdm(val_iter, desc=f'Epoch {epoch}/{n_epochs} (val)', leave=False):
            input = input.to(device)
            labels = [l.to(device) for l in labels]

            y_moved, y_dvf = model(input)
            x_fixed = input[:, 0].unsqueeze(1)  # Single-channel image should still have channel dimension.
            sim_loss = loss_sim(y_moved, x_fixed)
            smooth_loss = loss_smooth(y_dvf) 
            scaled_smooth_loss = loss_lambda * smooth_loss
            loss = sim_loss + scaled_smooth_loss

            # Record checkpointing metric.
            epoch_val_losses += [loss.item()]

            # Record metrics.
            if use_logging:
                run.log({
                    'epoch': epoch,
                    'step': step,
                    'val/loss': loss.item(),
                    'val/loss-sim': sim_loss.item(),
                    'val/loss-smooth': scaled_smooth_loss.item(),
                    'val/loss-smooth-raw': smooth_loss.item(),
                }, step=step)

                # Log images.
                if interval_matches(epoch, step, val_image_interval, len(val_iter)):
                    # Show coronal slices.
                    view = 1
                    slice_idx = input.shape[view + 2] // 2
                    batch_idx = 0
                    index = [batch_idx, 0, slice(None), slice(None), slice(None)]
                    index[view + 2] = slice_idx
                    x_f = input[index].cpu().numpy()
                    index[1] = 1
                    x_m = input[index].cpu().numpy()
                    index[1] = 0
                    y_m = y_moved[index].detach().cpu().numpy()
                    index[1] = slice(None)
                    y_d = y_dvf[index].detach().cpu().numpy()
                    y_d = np.moveaxis(y_d, 0, -1)
                    # Expect maximum DVF magnitudes of 0.25 times the image width, so normalise to [-0.5, 0.5) as
                    # as [-2, 2) represents moving across the whole image in 'grid_sample' coordintaes.
                    mag_min, mag_max = -0.5, 0.5
                    y_d = np.clip(y_d, mag_min, mag_max)
                    y_d = (y_d - mag_min) / (mag_max - mag_min)             # Shift to RGB data range.
                    # Convert to RGB image and show 'red' voxels for CT intensities below pad threshold.
                    pad_threshold_norm = (pad_threshold - norm_params['mean']) / norm_params['std'] if pad_threshold is not None else None
                    x_f = convert_to_rgb(x_f, pad_threshold_norm)
                    x_m = convert_to_rgb(x_m, pad_threshold_norm)
                    y_m = convert_to_rgb(y_m, pad_threshold_norm)
                    img = np.concatenate([x_m, x_f, y_m, y_d], axis=0)       # Create single image.
                    img = np.transpose(img)         # Wandb expects rows first.
                    img = np.moveaxis(img, 0, -1)   # Move RGB channels to last dimension.
                    img = np.flip(img, axis=0)      # Wandb plots rows from top down.

                    # Send image.
                    caption = f'{desc}:{get_axis_name(view)}'
                    image = wandb.Image(
                        img,
                        caption=caption,
                    )
                    run.log({ caption: image }, step=step)

        # Save mean validation loss.
        mean_val_loss = np.mean(epoch_val_losses)
        run.log({ 'val/loss-epoch-mean': mean_val_loss }, step=step)
        val_losses.append(mean_val_loss)

        # Save best model/s.
        if len(val_losses) >= val_loss_smoothing:
            smoothed_val_loss = np.mean(val_losses[-val_loss_smoothing:])
            run.log({ 'val/loss-ma': smoothed_val_loss }, step=step)
            if smoothed_val_loss < min_val_loss:
                min_val_loss = smoothed_val_loss

                # Update 'best_ckpts'.
                ckpt = f'loss={min_val_loss:.6f}_epoch={epoch}_step={step}'
                best_ckpts.insert(0, ckpt)
                if len(best_ckpts) > n_best_ckpts:
                    old_ckpt = best_ckpts.pop()
                    filepath = os.path.join(ckpt_path, f'{old_ckpt}.ckpt')
                    os.remove(filepath)

                # Save model.
                print(best_ckpts)
                ckpt_data = { 
                    'best-ckpts': best_ckpts,
                    'epoch': epoch,
                    'min-val-loss': min_val_loss,
                    'model': model.state_dict(),
                    'norm-params': norm_params,   # More reliable than writing to a file.
                    'optimiser': optimiser.state_dict(),
                    'step': step,
                    'val-losses': val_losses,     # Required for moving average.
                }
                filepath = os.path.join(ckpt_path, f'{ckpt}.ckpt')
                torch.save(ckpt_data, filepath)

        # Save current model.
        ckpt_data = { 
            'best-ckpts': best_ckpts,
            'epoch': epoch,
            'min-val-loss': min_val_loss,
            'model': model.state_dict(),
            'norm-params': norm_params,   # More reliable than writing to a file.
            'optimiser': optimiser.state_dict(),
            'step': step,
            'val-losses': val_losses,     # Required for moving average.
        }
        filepath = os.path.join(ckpt_path, 'last.ckpt')
        torch.save(ckpt_data, filepath)
    
def convert_to_rgb(
    data: np.ndarray,
    pad_threshold: Optional[float] = None) -> np.ndarray:
    min, max = data.min(), data.max()

    # Convert data to RGB format.
    data = (data - min) / (max - min)
    data = np.expand_dims(data, -1).repeat(3, axis=-1)

    if pad_threshold is not None:
        # Normalise the threshold to the new data range.
        pad_threshold = (pad_threshold - min) / (max - min)

        # Set values below threshold to red.
        data[:, :, 0][data[:, :, 0] < pad_threshold] = 1
        data[:, :, 1][data[:, :, 1] < pad_threshold] = 0
        data[:, :, 2][data[:, :, 2] < pad_threshold] = 0

    return data
