from multiprocessing import freeze_support

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import *
import wandb

from augmed import Pipeline, RandomAffine, RandomElastic, RandomGaussianNoise
from mymi import config
from mymi import logging
from mymi.losses import HeatmapKLDivLoss
from mymi.models import get_model, load_model
from mymi.utils.cdog import load_tiff_arc
from mymi.utils.interval import interval_matches

N_MARKERS = 3
MARKER_COLS = [f'Marker_{i}_{ax}' for i in range(N_MARKERS) for ax in ('X', 'Y')]

def gaussian_heatmaps(
    shape: Tuple[int, int],
    centers: torch.Tensor,   # (..., 2) [x, y] - any leading batch/channel dims.
    sigma: float,
    ) -> torch.Tensor:
    """Discrete 2D Gaussians centred at 'centers', each normalised to sum to 1 over the trailing (X, Y) dims."""
    device = centers.device
    xs = torch.arange(shape[0], device=device, dtype=torch.float32)
    ys = torch.arange(shape[1], device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(xs, ys, indexing='ij')   # X x Y.
    cx = centers[..., 0].unsqueeze(-1).unsqueeze(-1)
    cy = centers[..., 1].unsqueeze(-1).unsqueeze(-1)
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    g = g + 1e-12   # Avoid exact zeros (float32 underflow) which break KL divergence.
    return g / g.sum(dim=(-2, -1), keepdim=True)

def _augment_batch(
    xs: torch.Tensor,       # B x 1 x X x Y.
    points: torch.Tensor,   # B x n_markers x 2.
    transform: Pipeline,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies the augmed Pipeline per-sample (augmed doesn't support batched points) - the same random
    draw is applied consistently to each sample's image and its markers."""
    imgs_t, pts_t = [], []
    for i in range(xs.shape[0]):
        img_t, pts_i_t = transform(xs[i, 0], points[i])
        imgs_t.append(img_t)
        pts_t.append(pts_i_t)
    return torch.stack(imgs_t, dim=0).unsqueeze(1), torch.stack(pts_t, dim=0)

def _normalise_images(xs: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score normalisation. xs: B x 1 x X x Y."""
    mean = xs.mean(dim=(2, 3), keepdim=True)
    std = xs.std(dim=(2, 3), keepdim=True)
    return (xs - mean) / (std + 1e-8)

def _in_bounds_mask(
    points: torch.Tensor,   # B x n_markers x 2 [x, y].
    shape: Tuple[int, int],
    ) -> torch.Tensor:
    """True where a (possibly augmented) marker's coordinates lie within the image bounds - used to
    exclude off-grid markers (pred would be an uninformative edge/flat map) from the loss."""
    x_ok = (points[..., 0] >= 0) & (points[..., 0] < shape[0])
    y_ok = (points[..., 1] >= 0) & (points[..., 1] < shape[1])
    return x_ok & y_ok   # B x n_markers.

def _spatial_softmax_np(logits: np.ndarray) -> np.ndarray:
    """logits: (C, X, Y) raw model output -> per-channel probabilities summing to 1."""
    c = logits.shape[0]
    flat = logits.reshape(c, -1)
    flat = flat - flat.max(axis=1, keepdims=True)
    exp = np.exp(flat)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.reshape(logits.shape)

def _soft_argmax_coords(heatmap: np.ndarray) -> np.ndarray:
    """heatmap: (C, X, Y) normalised probabilities (sum=1 per channel) -> (C, 2) [x, y] sub-pixel
    coordinates, computed as the spatial expectation (centre of mass) of each channel's distribution."""
    _, x, y = heatmap.shape
    xs = np.arange(x, dtype=np.float32)
    ys = np.arange(y, dtype=np.float32)
    px = heatmap.sum(axis=2)   # C x X - marginal over y.
    py = heatmap.sum(axis=1)   # C x Y - marginal over x.
    coords = np.stack([(px * xs).sum(axis=1), (py * ys).sum(axis=1)], axis=-1)
    return coords.astype(np.float32)

def _plot_marker_predictions(
    x_slice: np.ndarray,
    gt_coords: np.ndarray,
    pred_coords: np.ndarray,
    pred_heatmap: np.ndarray,
    ) -> plt.Figure:
    """Input image with GT (green) / predicted (red) marker overlays, plus the max-over-channels predicted heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x_slice.T, cmap='gray', origin='lower')
    axes[0].scatter(gt_coords[:, 0], gt_coords[:, 1], c='lime', marker='o', s=40, label='GT')
    axes[0].scatter(pred_coords[:, 0], pred_coords[:, 1], c='red', marker='x', s=40, label='Pred')
    axes[0].set_title('Input + Markers')
    axes[0].legend(loc='upper right', fontsize='small')
    axes[0].axis('off')

    axes[1].imshow(pred_heatmap.T, cmap='viridis', origin='lower')
    axes[1].set_title('Predicted Heatmap (max over channels)')
    axes[1].axis('off')
    fig.tight_layout()
    return fig

def _log_marker_image(
    xs: torch.Tensor,
    ys: torch.Tensor,
    ys_pred: torch.Tensor,
    epoch: int,
    step: int,
    save_dir: str,
    log_local: bool,
    run,
    tag: str,
    batch_idx: int = 0,
    ) -> None:
    x_slice = xs[0, 0].cpu().numpy()
    gt_heatmap = ys[0].cpu().numpy()
    pred_probs = _spatial_softmax_np(ys_pred[0].detach().cpu().numpy())

    gt_coords = _soft_argmax_coords(gt_heatmap)
    pred_coords = _soft_argmax_coords(pred_probs)
    pred_heatmap = pred_probs.max(axis=0)

    fig = _plot_marker_predictions(x_slice, gt_coords, pred_coords, pred_heatmap)

    # 'step' doesn't advance during validation (it's frozen at the training step count for the
    # epoch), so 'batch_idx' is required to avoid every validation image overwriting the same file.
    filename = f'epoch-{epoch:04d}-step-{step:06d}-batch-{batch_idx:03d}.png'
    if log_local:
        fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    elif run is not None:
        run.log({f'{tag}/image': wandb.Image(fig)}, step=step)
    plt.close(fig)

class MarkerHeatmapDataset(Dataset):
    """Wraps in-memory kV frames + marker points. Augmentation, normalisation and heatmap rendering all
    happen per-batch in the training loop (on GPU) - so this just returns the raw image and raw points."""
    def __init__(
        self,
        images: np.ndarray,    # (N, X, Y) float32.
        points: np.ndarray,    # (N, n_markers, 2) float32, [X, Y] per marker.
        indices: Optional[np.ndarray] = None,   # Subset of frames this dataset exposes (e.g. train/val split).
        ) -> None:
        self.__images = images
        self.__points = points
        self.__indices = indices if indices is not None else np.arange(len(images))

    def __len__(self) -> int:
        return len(self.__indices)

    def __getitem__(
        self,
        index: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.__indices[index]
        image = self.__images[real_idx].astype(np.float32)
        x = torch.from_numpy(image[None])   # 1 x X x Y.

        points = torch.from_numpy(self.__points[real_idx].astype(np.float32))   # n_markers x 2.

        return x, points

def _derive_csv_dir(fraction_path: str) -> str:
    """Derive the manual-segmentation CSV directory from a '.../RNSH/<...>/<Patient>/<Fraction>' npz fraction path,
    e.g. '.../RNSH/Treatment files/Patient01/Fx01' -> '.../RNSH/MilesManualMarkerSegmentation/Patient01/Fx01'."""
    fx_id = os.path.basename(fraction_path)
    pat_id = os.path.basename(os.path.dirname(fraction_path))
    rnsh_dir = os.path.dirname(os.path.dirname(os.path.dirname(fraction_path)))
    return os.path.join(rnsh_dir, 'MilesManualMarkerSegmentation', pat_id, fx_id)

def load_valkim_arc_data(
    fractions: List[Tuple[str, List[int]]],   # [(fraction_path, arcs), ...] - one or more fractions/patients.
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate (image, marker-point) pairs across all given fractions/arcs, asserting
    consistent lengths, no NaNs, and a common image shape across fractions."""
    datas, pointses = [], []
    for fraction_path, arcs in fractions:
        csv_dir = _derive_csv_dir(fraction_path)
        for a in arcs:
            data, _ = load_tiff_arc(fraction_path, a)

            f = os.path.join(csv_dir, f'Arc{a:02d}', 'MarkerLocationsGA_CouchShift_adapted_edited.txt')
            df = pd.read_csv(f)
            assert len(df) == len(data), f"{fraction_path} arc {a}: image frames ({len(data)}) != marker CSV rows ({len(df)})."

            points = df[MARKER_COLS].to_numpy(dtype=np.float32)
            assert not np.isnan(points).any(), f"{fraction_path} arc {a}: marker CSV contains NaN values."
            points = points.reshape(-1, N_MARKERS, 2)
            points[:, :, 1] = data.shape[2] - points[:, :, 1]   # Y-coord needs to be flipped.

            datas.append(data)
            pointses.append(points)

    shapes = {d.shape[1:] for d in datas}
    assert len(shapes) == 1, f"Expected a common image shape across all fractions/arcs, got {shapes}."

    data = np.concatenate(datas, axis=0)
    points = np.concatenate(pointses, axis=0)
    return data, points

def _run_lr_find(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    project: str,
    model_name: str,
    shape: Tuple[int, int],
    sigma: float,
    transform: Optional[Pipeline] = None,
    min_lr: float = 1e-7,
    max_lr: float = 1,
    n_iter: int = 100,
    ) -> None:
    """Exponential LR range test - saves loss-vs-lr data/plot to help choose 'lr_init'."""
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    optimiser = torch.optim.AdamW(model.parameters(), lr=min_lr)
    mult = (max_lr / min_lr) ** (1 / n_iter)
    lr = min_lr

    lrs, losses = [], []
    best_loss = float('inf')
    model.train()
    train_iter = iter(train_loader)

    for i in tqdm(range(n_iter), desc='LR Find'):
        try:
            xs, points = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xs, points = next(train_iter)
        xs, points = xs.to(device), points.to(device)
        if transform is not None:
            xs, points = _augment_batch(xs, points, transform)
        mask = _in_bounds_mask(points, shape)
        xs = _normalise_images(xs)
        ys = gaussian_heatmaps(shape, points, sigma)

        ys_pred = model(xs)
        loss = loss_fn(ys_pred, ys, mask=mask)
        lrs.append(lr)
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
        if loss.item() > 10 * best_loss and i > 5:
            logging.info(f"Stopping LR find early at iter {i} - loss diverged.")
            break

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        lr *= mult
        for param_group in optimiser.param_groups:
            param_group['lr'] = lr

    model.load_state_dict(init_state)

    save_dir = os.path.join(config.dirs.lr_find, project, model_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'lr_find.json'), 'w') as f:
        json.dump({'lr': lrs, 'loss': losses}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lrs, losses)
    ax.set_xscale('log')
    ax.set_xlabel('Learning rate')
    ax.set_ylabel('KL-divergence loss')
    ax.set_title('LR Find')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'lr_find.png'), dpi=150)
    plt.close(fig)
    logging.info(f"Saved LR find results to '{save_dir}'.")

def train_markerseg(
    fractions: List[Tuple[str, List[int]]],   # [(fraction_path, arcs), ...] - one or more fractions/patients.
    project: str,
    model: str,
    n_epochs: int = 20,
    lr_init: float = 5e-4,
    arch: str = 'unet2d:m',
    batch_size: int = 32,
    sigma: float = 5.0,
    val_split: float = 0.2,
    log_images: bool = False,
    log_images_local: bool = True,
    n_train_images: int = 5,
    n_val_images: int = 5,
    train_image_interval: str = 'epoch:5',
    val_image_interval: str = 'epoch:5',
    lr_find: bool = False,
    lr_find_min_lr: float = 1e-7,
    lr_find_max_lr: float = 1,
    lr_find_n_iter: int = 100,
    num_workers: int = 4,
    random_seed: int = 42,
    resume: bool = False,
    resume_ckpt: str = 'last',
    use_logging: bool = True,
    overwrite: bool = False,
    ) -> None:
    model_name = model
    torch.manual_seed(random_seed)

    # Load data.
    data, points = load_valkim_arc_data(fractions)
    logging.info(f"Loaded {len(data)} frames across {len(fractions)} fraction(s): {[f for f, _ in fractions]}.")
    shape = data.shape[1:]   # (X, Y) - needed to render Gaussian heatmaps from (possibly augmented) points.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', f"Expected CUDA device, got '{device}'. Ensure a GPU is available."

    # Random affine + elastic augmentation, applied identically to the image and its markers.
    # Runs on GPU, per-sample, in the training loop (applied to already-batched, already-.to(device) tensors).
    # dim=2 must be set explicitly on each sub-transform - Pipeline's dim inference (from the points
    # shape) only sets its own dim, it doesn't cascade to sub-transforms, which otherwise default to dim=3.
    train_transform = Pipeline([
        RandomAffine(r=5, s=[0.8, 1.2], t=50, dim=2),
        RandomElastic(dim=2),
        RandomGaussianNoise(sp=0.01),
    ], device=device, fill='border', filter_offgrid=False)   # Replicate edge pixels - more realistic than a flat min-value fill.
    # filter_offgrid=False: markers must always number N_MARKERS per sample for heatmap rendering/batching - keep them even if pushed off-grid.

    # Random per-frame train/val split (index-only - augmentation is applied to the train split only).
    rng = np.random.default_rng(random_seed)
    perm = rng.permutation(len(data))
    n_val = int(len(data) * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_dataset = MarkerHeatmapDataset(data, points, indices=train_idx)
    val_dataset = MarkerHeatmapDataset(data, points, indices=val_idx)
    tl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    vl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f'[train] n_train={len(train_idx)} n_val={len(val_idx)} train_loader_len={len(tl)} val_loader_len={len(vl)}')

    ckpt_path = os.path.join(config.dirs.models, project, model_name)
    run_dir = os.path.join(config.dirs.runs, project, model_name)
    if not resume and not lr_find:
        existing = [p for p in (ckpt_path, run_dir) if os.path.exists(p)]
        if existing:
            if not overwrite:
                raise FileExistsError(f"Model/run '{model_name}' (project '{project}') already exists at {existing}. "
                    "Pass overwrite=True to replace it, or resume=True to continue training it.")
            for p in existing:
                logging.info(f"Removing old directory '{p}'.")
                shutil.rmtree(p)
    os.makedirs(ckpt_path, exist_ok=True)

    # Create model. Disable channel-wise softmax - each marker channel is an independent
    # spatial distribution, normalised via spatial softmax inside 'HeatmapKLDivLoss'.
    net = get_model(arch, N_MARKERS, use_softmax=False)
    ckpt_info = {}
    if resume:
        net, _, ckpt_info = load_model(net, project, model_name, resume_ckpt, device=device, state='train')
    else:
        net.to(device)

    loss_fn = HeatmapKLDivLoss()

    if lr_find:
        _run_lr_find(net, tl, loss_fn, device, project, model_name, shape, sigma, transform=train_transform, min_lr=lr_find_min_lr, max_lr=lr_find_max_lr, n_iter=lr_find_n_iter)
        return

    optimiser = torch.optim.AdamW(net.parameters(), lr=lr_init)
    if resume:
        logging.info(f"Restoring optimiser state from checkpoint {resume_ckpt}.")
        optimiser.load_state_dict(ckpt_info['optimiser'])

    if use_logging:
        run = wandb.init(dir=run_dir, entity='clarkbab', project=project, name=model_name)
    else:
        run = None

    train_image_save_dir = os.path.join(run_dir, 'images', 'train')
    val_image_save_dir = os.path.join(run_dir, 'images', 'val')
    if log_images_local:
        os.makedirs(train_image_save_dir, exist_ok=True)
        os.makedirs(val_image_save_dir, exist_ok=True)

    start_epoch = ckpt_info['epoch'] + 1 if resume else 0
    step = ckpt_info['step'] + 1 if resume else 0
    val_losses = ckpt_info.get('val-losses', []) if resume else []
    min_val_loss = ckpt_info.get('min-val-loss', np.inf) if resume else np.inf
    n_best_ckpts = 1
    best_ckpts = ckpt_info['best-ckpts'] if resume else []

    for e in range(start_epoch, n_epochs):
        # Training loop.
        net.train()
        for i, (xs, points) in enumerate(tqdm(tl, desc=f'Epoch {e}/{n_epochs} (train)', leave=False)):
            xs, points = xs.to(device), points.to(device)
            xs, points = _augment_batch(xs, points, train_transform)
            mask = _in_bounds_mask(points, shape)
            xs = _normalise_images(xs)
            ys = gaussian_heatmaps(shape, points, sigma)

            ys_pred = net(xs)
            loss = loss_fn(ys_pred, ys, mask=mask)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if use_logging:
                run.log({'epoch': e, 'step': step, 'train/loss': loss.item()}, step=step)

            if log_images and interval_matches(step, train_image_interval, len(tl), step_match_length=n_train_images):
                _log_marker_image(xs, ys, ys_pred, e, step, train_image_save_dir, log_images_local, run, 'train', batch_idx=i)

            step += 1

        # Validation loop.
        net.eval()
        epoch_val_losses = []
        for i, (xs, points) in enumerate(tqdm(vl, desc=f'Epoch {e}/{n_epochs} (val)', leave=False)):
            xs, points = xs.to(device), points.to(device)
            mask = _in_bounds_mask(points, shape)
            xs = _normalise_images(xs)
            ys = gaussian_heatmaps(shape, points, sigma)
            with torch.no_grad():
                ys_pred = net(xs)
                loss = loss_fn(ys_pred, ys, mask=mask)
            epoch_val_losses.append(loss.item())
            if use_logging:
                run.log({'epoch': e, 'step': step, 'val/loss': loss.item()}, step=step)

            if log_images and interval_matches(step, val_image_interval, len(tl)) and i < n_val_images:
                _log_marker_image(xs, ys, ys_pred, e, step, val_image_save_dir, log_images_local, run, 'val', batch_idx=i)

        mean_val_loss = float(np.mean(epoch_val_losses))
        val_losses.append(mean_val_loss)
        if use_logging:
            run.log({'val/loss-epoch': mean_val_loss}, step=step)

        # Save best model.
        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss

            ckpt = f'loss={min_val_loss:.6f}_epoch={e}_step={step}'
            best_ckpts.insert(0, ckpt)
            if len(best_ckpts) > n_best_ckpts:
                old_ckpt = best_ckpts.pop()
                os.remove(os.path.join(ckpt_path, f'{old_ckpt}.ckpt'))

            ckpt_data = {
                'best-ckpts': best_ckpts,
                'epoch': e,
                'min-val-loss': min_val_loss,
                'model': net.state_dict(),
                'optimiser': optimiser.state_dict(),
                'step': step,
                'val-losses': val_losses,
            }
            torch.save(ckpt_data, os.path.join(ckpt_path, f'{ckpt}.ckpt'))
            torch.save(ckpt_data, os.path.join(ckpt_path, 'best.ckpt'))

        # Save current model.
        ckpt_data = {
            'best-ckpts': best_ckpts,
            'epoch': e,
            'min-val-loss': min_val_loss,
            'model': net.state_dict(),
            'optimiser': optimiser.state_dict(),
            'step': step,
            'val-losses': val_losses,
        }
        torch.save(ckpt_data, os.path.join(ckpt_path, 'last.ckpt'))
        torch.save(ckpt_data, os.path.join(ckpt_path, f'epoch={e:04d}.ckpt'))

if __name__ == '__main__':
    freeze_support()

    fractions = [
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx01", [1, 2, 3]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx02", [1, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx03", [1, 2, 3]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient01\Fx04", [1, 2, 3]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx01", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx02", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx03", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx04", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx05", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx06", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx07", [1, 2, 3, 4]),
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient02\Fx08", [1, 2, 3, 4]),
    ]
    arch = 'unet2d:m'
    version = '007'
    model_name = f"markerseg-{arch.replace(':', '_')}-{version}"

    train_markerseg(
        fractions=fractions,
        project='MLM-VALKIM',
        model=model_name,
        n_epochs=20,
        lr_init=5e-4,
        arch=arch,
        batch_size=16,
        sigma=5.0,
        val_split=0.2,
        log_images=True,
        log_images_local=True,
        lr_find=False,
        num_workers=4,
        random_seed=42,
        resume=True,
        use_logging=True,
        overwrite=False,
    )


