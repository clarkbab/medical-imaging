import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from mymi import config
from mymi import logging
from mymi.models import get_model
from mymi.utils.cdog import load_tiff_arc

N_MARKERS = 3

def _spatial_softmax_np(logits: np.ndarray) -> np.ndarray:
    """logits: (B, C, X, Y) raw model output -> probabilities summing to 1 over the trailing (X, Y) dims."""
    b, c, x, y = logits.shape
    flat = logits.reshape(b, c, -1)
    flat = flat - flat.max(axis=-1, keepdims=True)
    exp = np.exp(flat)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    return probs.reshape(b, c, x, y)

def _soft_argmax_coords(heatmaps: np.ndarray) -> np.ndarray:
    """heatmaps: (B, C, X, Y) raw logits -> (B, C, 2) [x, y] sub-pixel coords of each channel's peak,
    computed as the spatial expectation (centre of mass) of the softmax probability map."""
    _, _, x, y = heatmaps.shape
    probs = _spatial_softmax_np(heatmaps)
    xs = np.arange(x, dtype=np.float32)
    ys = np.arange(y, dtype=np.float32)
    px = probs.sum(axis=3)   # B x C x X - marginal over y.
    py = probs.sum(axis=2)   # B x C x Y - marginal over x.
    coords = np.stack([(px * xs).sum(axis=-1), (py * ys).sum(axis=-1)], axis=-1)
    return coords.astype(np.float32)

def _save_predictions(
    coords: np.ndarray,
    fraction_path: str,
    arc: int,
    ) -> str:
    """Save predicted (n_frames, n_markers, 2) coords as a CSV mirroring the manual marker files, under
    '<RNSH>/MilesManualMarkerSegmentation/<Patient>/<Fraction>/Arc<n>/MarkerLocationsGA_CouchShift_adapted.txt'."""
    n_frames = coords.shape[0]
    # 'Frame No' is just the sequential frame index here - the original hardware frame counter
    # (as seen in the manually-segmented files) isn't recoverable from the processed npz.
    out_df = pd.DataFrame({'Frame No': np.arange(n_frames)})
    for i in range(N_MARKERS):
        out_df[f'Marker_{i}_X'] = coords[:, i, 0]
        out_df[f'Marker_{i}_Y'] = coords[:, i, 1]

    # Save under the manual-segmentation tree (sibling of the treatment-files root), e.g.
    # '.../RNSH/MilesManualMarkerSegmentation/Patient01/Fx02/Arc01/...', not next to the input npz.
    fx_id = os.path.basename(fraction_path)
    pat_id = os.path.basename(os.path.dirname(fraction_path))
    rnsh_dir = os.path.dirname(os.path.dirname(os.path.dirname(fraction_path)))
    out_dir = os.path.join(rnsh_dir, 'MilesManualMarkerSegmentation', pat_id, fx_id, f'Arc{arc:02d}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'MarkerLocationsGA_CouchShift_adapted.txt')
    out_df.to_csv(out_path, index=False)
    return out_path

def _predict_arc(
    net: torch.nn.Module,
    device: torch.device,
    fraction_path: str,
    arc: int,
    batch_size: int,
    ) -> np.ndarray:
    """Predict (n_frames, n_markers, 2) marker locations for a single arc, saving the result as a CSV."""
    data, _ = load_tiff_arc(fraction_path, arc)   # (n_frames, X, Y) float32 - same preprocessing as training.

    n_frames = data.shape[0]
    all_coords = []
    with torch.no_grad():
        desc = f'Predicting markers ({os.path.basename(fraction_path)} arc {arc})'
        for start in tqdm(range(0, n_frames, batch_size), desc=desc, leave=False):
            batch = data[start:start + batch_size]
            batch = (batch - batch.mean(axis=(1, 2), keepdims=True)) / (batch.std(axis=(1, 2), keepdims=True) + 1e-8)
            xs = torch.from_numpy(batch[:, None].astype(np.float32)).to(device)

            ys_pred = net(xs).cpu().numpy()   # B x n_markers x X x Y raw logits.
            all_coords.append(_soft_argmax_coords(ys_pred))
    coords = np.concatenate(all_coords, axis=0)   # (n_frames, n_markers, 2), [x, y] in image-array space.

    coords[:, :, 1] = data.shape[2] - coords[:, :, 1]   # Undo the Y-flip applied during training - back to CSV convention.

    out_path = _save_predictions(coords, fraction_path, arc)
    logging.info(f"Saved marker predictions for {n_frames} frames to '{out_path}'.")

    return coords

def predict_markerseg(
    fractions: List[Tuple[str, List[int]]],   # [(fraction_path, arcs), ...] - one or more fractions/patients.
    project: str,
    model: str,
    ckpt: str = 'best',
    arch: str = 'unet2d:m',
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    ) -> Dict[Tuple[str, int], np.ndarray]:
    """Predict marker locations for every (fraction_path, arc) pair, saving each as a CSV (Frame No +
    Marker_N_X/Marker_N_Y columns, matching the manual marker files)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model once from the literal '<ckpt>.ckpt' file (e.g. 'best.ckpt', saved by train_markerseg
    # on every new min val loss) - NOT via 'load_model's alphabetical alias resolution, which
    # assumes a higher-is-better metric and would pick the worst checkpoint for a loss metric.
    net = get_model(arch, N_MARKERS, use_softmax=False)
    ckpt_path = os.path.join(config.dirs.models, project, model, f'{ckpt}.ckpt')
    ckpt_data = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt_data['model'])
    net.to(device)
    net.eval()

    results = {}
    for fraction_path, arcs in fractions:
        for arc in arcs:
            results[(fraction_path, arc)] = _predict_arc(net, device, fraction_path, arc, batch_size)

    return results

if __name__ == '__main__':
    fractions = [
        (r"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient03\Fx02", [1, 2, 3]),
    ]
    arch = 'unet2d:m'
    version = '006'
    model_name = f"markerseg-{arch.replace(':', '_')}-{version}"

    predict_markerseg(
        fractions=fractions,
        project='MLM-VALKIM',
        model=model_name,
        ckpt='best',
        arch=arch,
    )
