from dicomset.utils import logger
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
import torch
from tqdm import tqdm

def save_lr_find(
    dataset: str,
    project: str,
    model_name: str,
    lrs: list,
    losses: list,
) -> None:
    """Save LR find results as JSON to <dataset>/images/<project>/<model_name>/."""
    from dicomset.training import TrainingDataset
    save_dir = os.path.join(TrainingDataset(dataset).path, 'images', project, model_name)
    os.makedirs(save_dir, exist_ok=True)
    results = {'lr': lrs, 'loss': losses}
    json_path = os.path.join(save_dir, 'lr_find.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved LR find data to '{json_path}'.")
    create_lr_find_plot(results['lr'], results['loss'], save_dir)


def plot_lr_find(
    dataset: str,
    project: str,
    model_name: str,
) -> None:
    """Display the saved LR find plot (for use in Jupyter notebooks)."""
    from dicomset.training import TrainingDataset
    from IPython.display import display, Image
    save_dir = os.path.join(TrainingDataset(dataset).path, 'images', project, model_name)
    plot_path = os.path.join(save_dir, 'lr_find.png')
    display(Image(plot_path))

def run_lr_find_seg(
    model: torch.nn.Module,
    train_loader,
    loss_fn,
    device: torch.device,
    dataset: str,
    project: str,
    model_name: str,
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
) -> None:
    """Run an LR range test and save results + plot to files/mlm/valkim/."""
    logger.log_method()

    # Save initial model state so we can restore it afterwards.
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Set up optimiser at minimum learning rate.
    optimiser = torch.optim.AdamW(model.parameters(), lr=min_lr)

    # Exponential schedule: lr_{i+1} = lr_i * mult.
    mult = (max_lr / min_lr) ** (1 / n_iter)
    lr = min_lr

    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    train_loader.dataset.create_projections(0)
    train_iter = iter(train_loader)

    for i in tqdm(range(n_iter), desc='LR Find'):
        # Get next batch (cycle through loader if exhausted).
        try:
            xs, ys, angles = next(train_iter)
        except StopIteration:
            train_loader.dataset.create_projections(0)
            train_iter = iter(train_loader)
            xs, ys, angles = next(train_iter)

        xs = xs.to(device)
        ys = ys.to(device)
        ys = ys[:, :2]  # GTV + background.

        # Forward pass.
        ys_pred = model(xs)
        loss = loss_fn(ys_pred, ys)

        # Record.
        lrs.append(lr)
        losses.append(loss.item())

        # Stop early if loss has diverged (> 10x the best recorded loss).
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 10 * best_loss and i > 5:
            logging.info(f"Stopping LR find early at iter {i} – loss diverged.")
            break

        # Backward pass + optimiser step.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update learning rate.
        lr *= mult
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    # Restore initial model weights.
    model.load_state_dict(init_state)

    save_lr_find(dataset, project, model_name, lrs, losses)


def run_lr_find_unet2d(
    model: torch.nn.Module,
    train_loader,
    loss_fn,
    device: torch.device,
    dataset: str,
    project: str,
    model_name: str,
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
    n_output_channels: int = 2,
) -> None:
    logger.log_method()

    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimiser = torch.optim.AdamW(model.parameters(), lr=min_lr)
    mult = (max_lr / min_lr) ** (1 / n_iter)
    lr = min_lr

    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    train_iter = iter(train_loader)

    for i in tqdm(range(n_iter), desc='LR Find (unet2d)'):
        try:
            xs, ys, angles = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xs, ys, angles = next(train_iter)

        xs = xs.to(device)
        ys = ys.to(device)
        ys = ys[:, :n_output_channels]

        ys_pred = model(xs)
        loss = loss_fn(ys_pred, ys)

        lrs.append(lr)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 10 * best_loss and i > 5:
            logger.info(f"Stopping LR find early at iter {i} – loss diverged.")
            break

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        lr *= mult
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    model.load_state_dict(init_state)
    save_lr_find(dataset, project, model_name, lrs, losses)


def run_lr_find_pix2pix(
    model,
    train_loader,
    device: torch.device,
    dataset: str,
    project: str,
    model_name: str,
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
    output_nc: int = 1,
) -> None:
    """Run an LR range test for a Pix2PixModel and save results + plot.

    Both the generator and discriminator are swept at the same learning rate.
    The G_L1 loss is used as the tracking metric since it best reflects
    segmentation quality.
    """
    logger.log_method()

    # Save initial state so we can restore afterwards.
    init_netG = copy.deepcopy(model.netG.state_dict())
    init_netD = copy.deepcopy(model.netD.state_dict())
    init_opt_G = copy.deepcopy(model.optimizer_G.state_dict())
    init_opt_D = copy.deepcopy(model.optimizer_D.state_dict())

    # Reset optimisers to min_lr.
    for param_group in model.optimizer_G.param_groups:
        param_group['lr'] = min_lr
    for param_group in model.optimizer_D.param_groups:
        param_group['lr'] = min_lr

    mult = (max_lr / min_lr) ** (1 / n_iter)
    lr = min_lr

    lrs = []
    losses = []
    best_loss = float('inf')

    model.netG.train()
    model.netD.train()
    # train_loader.dataset.create_projections(0)
    train_iter = iter(train_loader)

    for i in tqdm(range(n_iter), desc='LR Find (pix2pix)'):
        try:
            xs, ys, angles = next(train_iter)
        except StopIteration:
            # train_loader.dataset.create_projections(0)
            train_iter = iter(train_loader)
            xs, ys, angles = next(train_iter)

        input_dict = {
            'A': xs.to(device),
            'B': ys[:, :output_nc].to(device),
            'A_paths': [],
            'B_paths': [],
        }
        model.set_input(input_dict)
        model.optimize_parameters()

        loss_val = model.loss_G_L1.item()
        lrs.append(lr)
        losses.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
        if loss_val > 10 * best_loss and i > 5:
            logger.info(f"Stopping LR find early at iter {i} – loss diverged.")
            break

        # Advance learning rate.
        lr *= mult
        for param_group in model.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in model.optimizer_D.param_groups:
            param_group['lr'] = lr

    # Restore initial state.
    model.netG.load_state_dict(init_netG)
    model.netD.load_state_dict(init_netD)
    model.optimizer_G.load_state_dict(init_opt_G)
    model.optimizer_D.load_state_dict(init_opt_D)

    save_lr_find(dataset, project, model_name, lrs, losses)

def run_lr_find_bsp(
    model: torch.nn.Module,
    train_loader,
    loss_fn,
    device: torch.device,
    dataset: str,
    project: str,
    model_name: str,
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
) -> None:
    """Run an LR range test for the BSP model and save results + plot."""
    logger.log_method()

    # Save initial model state so we can restore it afterwards.
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Set up optimiser at minimum learning rate.
    optimiser = torch.optim.AdamW(model.parameters(), lr=min_lr)

    # Exponential schedule: lr_{i+1} = lr_i * mult.
    mult = (max_lr / min_lr) ** (1 / n_iter)
    lr = min_lr

    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    train_iter = iter(train_loader)

    for i in tqdm(range(n_iter), desc='LR Find (BSP)'):
        # Get next batch (cycle through loader if exhausted).
        try:
            xs, ys = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xs, ys = next(train_iter)

        xs = xs.to(device)
        ys = ys.to(device)

        # Forward pass.
        ys_pred = model(xs)
        phase = ys[:, 1, :, 1]  # (B, n_frames) in [0, 1]
        phase_sin = torch.sin(2 * torch.pi * phase)
        phase_cos = torch.cos(2 * torch.pi * phase)
        loss = (loss_fn(ys_pred[:, 0], ys[:, 0, :, 1])
                + loss_fn(ys_pred[:, 1], phase_sin)
                + loss_fn(ys_pred[:, 2], phase_cos))

        # Record.
        lrs.append(lr)
        losses.append(loss.item())

        # Stop early if loss has diverged (> 10x the best recorded loss).
        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 10 * best_loss and i > 5:
            logger.info(f"Stopping LR find early at iter {i} – loss diverged.")
            break

        # Backward pass + optimiser step.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update learning rate.
        lr *= mult
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    # Restore initial model weights.
    model.load_state_dict(init_state)

    save_lr_find(dataset, project, model_name, lrs, losses)


def create_lr_find_plot(
    lrs: list,
    losses: list,
    save_dir: str,
) -> None:
    """Plot loss vs learning rate and save to *save_dir*/<model_name>.png."""
    lrs = np.array(lrs)
    losses = np.array(losses)

    # Remove NaN losses.
    valid = ~np.isnan(losses)
    lrs = lrs[valid]
    losses = losses[valid]
    if len(lrs) == 0:
        logger.info("No valid loss values recorded – skipping plot.")
        return

    # Smooth for suggestion.
    smooth_losses = gaussian_filter1d(losses, sigma=2)

    # Suggestion: LR at steepest negative gradient.
    grad = np.gradient(smooth_losses)
    min_grad_idx = int(grad.argmin())
    sugg_lr = lrs[min_grad_idx]
    sugg_loss = losses[min_grad_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(lrs, losses, label='Loss')
    ax.plot(lrs, smooth_losses, label='Loss (smoothed)', alpha=0.7)
    ax.scatter([sugg_lr], [sugg_loss], color='red', zorder=5,
               label=f'Suggested LR: {sugg_lr:.2e}')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title('LR Find')
    ax.legend()
    ax.grid(True, which='major')
    ax.grid(True, which='minor', linestyle='--', alpha=0.4)

    # Minor log ticks.
    min_exp = int(np.floor(np.log10(np.min(lrs)))) - 1
    max_exp = int(np.ceil(np.log10(np.max(lrs)))) + 1
    minor_ticks = [c * 10 ** e for c in range(1, 10) for e in range(min_exp, max_exp)]
    ax.xaxis.set_minor_locator(FixedLocator(minor_ticks))

    plot_path = os.path.join(save_dir, 'lr_find.png')
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.info(f"Saved LR find plot to '{plot_path}'.")
    logger.info(f"Suggested learning rate: {sugg_lr:.2e}")
