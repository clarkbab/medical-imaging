import torch

from .contrast import contrast, plot_contrast, plot_contrast_hist
from .dice import batch_mean_dice, dice
from .distances import all_distances, apl, batch_mean_all_distances, extent_centre_distance, get_encaps_dist_mm, get_encaps_dist_vox, hausdorff_distance, mean_surface_distance, surface_dice, surface_distances, distances_deepmind
from .volume import volume

# In which direction does the metric improve?
# Higher is better (True) or lower is better (False).
def higher_is_better(metric: str) -> bool:
    if 'apl-mm-tol-' in metric:
        return False
    if metric == 'dice':
        return True
    if 'dm-surface-dice-tol-' in metric:
        return True
    if 'hd' in metric: 
        return False
    if metric == 'msd':
        return False
