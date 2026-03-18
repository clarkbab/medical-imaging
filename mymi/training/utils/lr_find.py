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

from mymi import config
from mymi import logging


def run_lr_find(
    model: torch.nn.Module,
    train_loader,
    loss_fn,
    device: torch.device,
    model_name: str,
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
) -> None:
    """Run an LR range test and save results + plot to files/mlm/valkim/."""
    logging.info(f"Running LR find: min_lr={min_lr}, max_lr={max_lr}, n_iter={n_iter}")

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

    # --- Save results ---
    save_dir = os.path.join(config.directories.files, 'mlm', 'valkim', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save data as JSON.
    results = {'lr': lrs, 'loss': losses}
    json_path = os.path.join(save_dir, 'lr-find.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved LR find data to '{json_path}'.")

    # --- Plot ---
    _plot_lr_find(lrs, losses, save_dir)


def _plot_lr_find(
    lrs: list,
    losses: list,
    save_dir: str,
) -> None:
    """Plot loss vs learning rate and save to *save_dir*/lr-find.png."""
    lrs = np.array(lrs)
    losses = np.array(losses)

    # Remove NaN losses.
    valid = ~np.isnan(losses)
    lrs = lrs[valid]
    losses = losses[valid]
    if len(lrs) == 0:
        logging.info("No valid loss values recorded – skipping plot.")
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

    plot_path = os.path.join(save_dir, 'lr-find.png')
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logging.info(f"Saved LR find plot to '{plot_path}'.")
    logging.info(f"Suggested learning rate: {sugg_lr:.2e}")
