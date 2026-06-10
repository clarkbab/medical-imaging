import dicomset as ds
from dicomset.utils import logger, plot_slice, load_numpy
from dicomset.typing import *
from augmed import Pipeline, Standardise, config as amconf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
from tqdm import tqdm
from typing import *

from mymi.models.architectures.bsp import BSPUNet2DPhase
from mymi.models.models import load_model

def create_bsp_predictions() -> None:
    dataset = 'VALKIM-BSP'
    project = 'BSP-VALKIM'
    model = 'unet2dphase-006'
    ckpt = 'best'

    set = ds.load(dataset, 'training')
    testpath = os.path.join(set.path, 'data', 'test')

    # Output directory for images.
    image_dir = os.path.join(testpath, 'images')
    os.makedirs(image_dir, exist_ok=True)

    # Load model.
    net, device, _ = load_model(BSPUNet2DPhase, project, model, ckpt)

    # Build preprocessing pipeline (matches validation pipeline used during training).
    amconf.set_dim(2)
    pipe = Pipeline([Standardise()], device=device)

    # Enumerate test files.
    test_files = sorted([f for f in os.listdir(testpath) if f.endswith('.npz') and not f.endswith('_pred.npz')])
    logger.info(f"Found {len(test_files)} test samples in '{testpath}'.")

    for fname in tqdm(test_files, desc='Predicting'):
        filepath = os.path.join(testpath, fname)
        shroud, signals = load_numpy(filepath, keys=['shroud', 'signals'])
        # shroud:  (n_frames, y_size)
        # signals: (2, n_frames, 2) — [amplitude, phase] as (n_frames, 2) points with y in [0, 1]

        n_frames = shroud.shape[0]
        xs_pts = np.arange(n_frames, dtype=float)

        # Standardise shroud using AugMed pipeline and build input tensor (1, 1, n_frames, y_size).
        shroud_std = pipe(shroud)
        x = torch.from_numpy(shroud_std).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(x)  # (1, 3, n_frames)

        # Decode amplitude and phase.
        amp_pred = pred[0, 0].cpu().numpy()                                              # (n_frames,)
        phase_pred = (np.arctan2(pred[0, 1].cpu().numpy(),
                                 pred[0, 2].cpu().numpy()) / (2 * np.pi)) % 1.0          # (n_frames,)

        # Build predicted signals as (2, n_frames, 2) points.
        signals_pred = np.stack([
            np.stack([xs_pts, amp_pred], axis=1),
            np.stack([xs_pts, phase_pred], axis=1),
        ], axis=0)

        # Save predictions.
        sample_name = os.path.splitext(fname)[0]
        np.savez(os.path.join(testpath, f'{sample_name}_pred.npz'), signals=signals_pred)

        # --- Plot GT vs predicted ---
        gt_pts_list = [signals[0].copy(), signals[1].copy()]
        pred_pts_list = [signals_pred[0].copy(), signals_pred[1].copy()]

        # Normalise y-coords to pixel space for plotting.
        gt_pts_list   = [normalise_points(p, min=0, max=shroud.shape[1] - 1) for p in gt_pts_list]
        pred_pts_list = [normalise_points(p, min=0, max=shroud.shape[1] - 1) for p in pred_pts_list]

        fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        plot_slice(shroud, ax=axs[0], points=gt_pts_list, title='GT', aspect=0.1, orientation='LI')
        plot_slice(shroud, ax=axs[1], points=pred_pts_list, title='Pred', aspect=0.1, orientation='LI')

        # Difference plot (pred - GT) for amplitude and phase.
        palette = sns.color_palette('colorblind')
        amp_diff = signals_pred[0][:, 1] - signals[0][:, 1]
        phase_diff = (signals_pred[1][:, 1] - signals[1][:, 1] + 0.5) % 1.0 - 0.5
        axs[2].scatter(xs_pts, amp_diff, color=palette[0], label='Amplitude diff', s=20)
        axs[2].scatter(xs_pts, phase_diff, color=palette[1], label='Phase diff', s=20)
        axs[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
        axs[2].set_title('Diff')
        axs[2].set_xlabel('Frame')
        axs[2].set_ylim(-1, 1)
        axs[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(image_dir, f'{sample_name}_pred.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    logger.info(f"Predictions saved to '{testpath}'.")
    logger.info(f"Images saved to '{image_dir}'.")

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
    create_bsp_predictions()
