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
from mymi.models import get_module, load_model
from mymi.utils import *

def train_segmentation(
    dataset: str,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    arch: str = 'unet3d:m',
    loss_fn: Literal['dice'] = 'dice',
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
    device = torch.device('cuda')
    n_output_channels = len(region_ids) + 1
    module = get_module(arch, n_output_channels)
    if resume:
        model, _, ckpt_info = load_model(module, project, model_name, resume_ckpt, device=device, state='train')
        if ckpt_info['norm-params'] != norm_params:
            raise ValueError(f"Got different norm params for loader {norm_params} and model {ckpt_info['norm-params']}.") 
    else:
        model = module()
        model.to(device)

    # Define loss function.
    if loss_fn == 'dice':
        loss_fn = DiceLoss()
    else:
        raise ValueError(f"Unknown loss function '{loss_fn}'.")
    logging.info(f"Using loss {loss_fn}.")

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
        for desc, x, ys, masks in tqdm(train_iter, desc=f'Epoch {epoch}/{n_epochs} (train)', leave=False):
            x = x.to(device)
            ys = [l.to(device) for l in ys]
            masks = [m.to(device) for m in masks]
            assert len(ys) == 1, f"Expected single label for segmentation task, got {len(ys)}."
            assert len(masks) == 1, f"Expected single mask for segmentation task, got {len(masks)}."

            # Perform training update.
            y_pred = model(x)
            y = ys[0]
            loss = loss_fn(y_pred, y, mask=masks[0])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Record metrics.
            if use_logging:
                run.log({
                    'epoch': epoch,
                    'step': step,
                    'train/loss': loss.item(),
                }, step=step)

            # Increment training step.
            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        for desc, x, ys, masks in tqdm(val_iter, desc=f'Epoch {epoch}/{n_epochs} (val)', leave=False):
            x = x.to(device)
            ys = [l.to(device) for l in ys]
            masks = [m.to(device) for m in masks]
            assert len(ys) == 1, f"Expected single label for segmentation task, got {len(ys)}."
            assert len(masks) == 1, f"Expected single mask for segmentation task, got {len(masks)}."

            # Make prediction.
            y_pred = model(x)
            y = ys[0]
            loss = loss_fn(y_pred, y, mask=masks[0])

            # Record checkpointing metric.
            epoch_val_losses += [loss.item()]

            # Record metrics.
            if use_logging:
                run.log({
                    'epoch': epoch,
                    'step': step,
                    'val/loss': loss.item(),
                }, step=step)

                # Log images.
                if interval_matches(epoch, step, val_image_interval, len(val_iter)):
                    for i, r in enumerate(region_ids):
                        channel = i + 1

                        # Skip channel if GT not present.
                        if not mask[i, c]:
                            continue

                        # Get region images.
                        y_pred_r, x_r, y_r = y_pred[i, c], x[i, 0], y[i, c]

                        # Get centre of extent of ground truth.
                        centre = fov_centre(y_r)
                        if centre is None:
                            # Presumably data augmentation has pushed the label out of view.
                            continue

                        # Plot each orientation.
                        for a, c in enumerate(centre):
                            # Get 2D slice.
                            indices = tuple([c if k == a else slice(0, x_r.shape[i]) for k in range(3)])
                            y_pred_slice, x_slice, y_slice = y_pred_r[indices], x_r[indices], y_r[indices]

                            # Fix orientation.
                            if a in (0, 1):     # Sagittal/coronal views.
                                y_pred_slice = np.rot90(y_pred_slice)
                                x_slice = np.rot90(x_slice)
                                y_slice = np.rot90(y_slice)
                            else:               # Axial view.
                                x_slice = np.transpose(x_slice)
                                y_slice = np.transpose(y_slice)
                                y_pred_slice = np.transpose(y_pred_slice)

                            # Send image.
                            title = f'desc:{d}:region:{r}:axis:{a}'
                            caption = d,
                            masks = {
                                'ground_truth': {
                                    'mask_data': y_slice,
                                    'class_labels': class_labels
                                },
                                'predictions': {
                                    'mask_data': y_pred_slice,
                                    'class_labels': class_labels
                                }
                            }
                            run.log({
                                title: wandb.Image(
                                    x_slice,
                                    caption=caption,
                                    masks=masks,
                                )
                            }, step=step)

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
