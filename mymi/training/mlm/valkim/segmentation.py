from augmed import Pipeline, RandomAffine
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
from typing import *
import wandb

from mymi import config
from mymi.geometry import fov_centre
from mymi.loaders import DRRLoader
from mymi import logging
from mymi.losses import DiceLoss
from mymi.models import get_model, load_model
from mymi.utils import *

def train_segmentation(
    dataset: str,
    pat: PatientID,
    project: str,
    model: str,
    n_epochs: int,
    lr_init: float,
    arch: str = 'unet3d:m',
    batch_size: int = 32,
    log_images: bool = False,
    loss_fn: Literal['dice'] = 'dice',
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    use_logging: bool = True,
    val_image_interval: str = 'epoch:5',
    ) -> None:
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

    # Create augmentation.
    transform_train = Pipeline([
        RandomAffine(r=10, s=[0.8, 1.2], t=20),
    ])
    print(transform_train)

    # Create data loaders.
    loader_kwargs = dict(
        batch_size=batch_size,
        transform_train=transform_train,
    )
    tl, vl = DRRLoader.build_loaders(dataset, pat, **loader_kwargs)

    # Create model.
    device = torch.device('cuda')
    n_output_channels = 2   # GTV plus background.
    model = get_model(arch, n_output_channels)
    if resume:
        model, _, ckpt_info = load_model(model, project, model_name, resume_ckpt, device=device, state='train')
    else:
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
    for e in range(start_epoch, n_epochs):
        # Training loop.
        model.train()
        train_iter = iter(tl)
        train_iter.create_projections(e)
        for xs, ys in tqdm(train_iter, desc=f'Epoch {e}/{n_epochs} (train)', leave=False):
            xs = xs.to(device)
            ys = ys.to(device)

            # Perform training update.
            ys_pred = model(xs)
            loss = loss_fn(ys_pred, ys)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Record metrics to Wandb.
            if use_logging:
                run.log({
                    'epoch': e,
                    'step': step,
                    'train/loss': loss.item(),
                }, step=step)

            # Increment training step.
            step += 1

        # Validation loop.
        model.eval()
        val_iter = iter(vl)
        epoch_val_losses = []
        for xs, ys in tqdm(val_iter, desc=f'Epoch {e}/{n_epochs} (val)', leave=False):
            xs = xs.to(device)
            ys = ys.to(device)

            # Make prediction.
            ys_pred = model(xs)
            loss = loss_fn(ys_pred, ys)

            # Record checkpointing metric.
            epoch_val_losses += [loss.item()]

            # Record metrics.
            if use_logging:
                run.log({
                    'epoch': e,
                    'step': step,
                    'val/loss': loss.item(),
                }, step=step)

                # Log images.
                if log_images and interval_matches(e, step, val_image_interval, len(val_iter)):
                    regions = ['GTV']
                    for i, r in enumerate(regions):
                        c = i + 1

                        # Log first batch item only.
                        x_r = xs[0].cpu().numpy()
                        y_r = ys[0, i + 1].cpu().numpy()   # First channel is background.
                        y_pred_r = ys_pred[0, i + 1].cpu().numpy()

                        # Get centre of extent of ground truth.
                        centre_vox = fov_centre(y_r)
                        if centre_vox is None:
                            # Presumably data augmentation has pushed the label out of view.
                            continue

                        # Plot each orientation.
                        for a, c in enumerate(centre_vox):
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
                            title = f'region:{r}:axis:{a}'
                            masks = {
                                'ground_truth': {
                                    'mask_data': y_slice,
                                    'class_labels': regions,
                                },
                                'predictions': {
                                    'mask_data': y_pred_slice,
                                    'class_labels': regions,
                                }
                            }
                            run.log({
                                title: wandb.Image(
                                    x_slice,
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
                ckpt = f'loss={min_val_loss:.6f}_epoch={e}_step={step}'
                best_ckpts.insert(0, ckpt)
                if len(best_ckpts) > n_best_ckpts:
                    old_ckpt = best_ckpts.pop()
                    filepath = os.path.join(ckpt_path, f'{old_ckpt}.ckpt')
                    os.remove(filepath)

                # Save model.
                print(best_ckpts)
                ckpt_data = { 
                    'best-ckpts': best_ckpts,
                    'epoch': e,
                    'min-val-loss': min_val_loss,
                    'model': model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'step': step,
                    'val-losses': val_losses,     # Required for moving average.
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
            'val-losses': val_losses,     # Required for moving average.
        }
        filepath = os.path.join(ckpt_path, 'last.ckpt')
        torch.save(ckpt_data, filepath)
