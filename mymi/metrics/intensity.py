import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import SimpleITK as sitk
from typing import List, Tuple

from mymi import types

# Calculate signal-to-noise ratio.
def snr(
    image: np.ndarray,
    label: np.ndarray,
    brain_label: np.ndarray,
    spacing: types.Spacing3D,
    d: float = 3) -> float:
    if image.shape != label.shape or image.shape != brain_label.shape:
        raise ValueError(f"Metric 'snr' expects images of equal shape. Got '{image.shape}', '{label.shape}', and '{brain_label.shape}'.")
    if (image.dtype != np.float32 and image.dtype != np.float64) or label.dtype != np.bool_ or brain_label.dtype != np.bool_:
        raise ValueError(f"Metric 'snr' expects (float, boolean, boolean) images. Got '{image.dtype}', '{label.dtype}', and '{brain_label.dtype}'.")
    if label.sum() == 0 or brain_label.sum() == 0:
        raise ValueError(f"Metric 'snr' can't be calculated with an empty 'label' or 'brain_label'. Got sums '{label.sum()}' and '{brain_label.sum()}'.")

    # Get structure mean and std. deviation.
    struct_vals = image[np.nonzero(label)]
    struct_mean = struct_vals.mean()
    struct_std = struct_vals.std()

    # Calculate margin of 'd' mm around structure. 
    label_itk = label.astype('uint8')
    label_itk = sitk.GetImageFromArray(label_itk)
    label_itk.SetSpacing(tuple(reversed(spacing)))
    label_dist_map = sitk.SignedMaurerDistanceMap(label_itk, useImageSpacing=True, squaredDistance=False, insideIsPositive=False)
    label_dist_map = sitk.GetArrayFromImage(label_dist_map)
    margin = np.zeros_like(label, dtype=bool)
    margin[(label_dist_map > 0) & (label_dist_map <= d)] = 1

    # Get margin mean and std. deviation.
    margin_vals = image[margin == 1]
    margin_mean = margin_vals.mean()
    margin_std = margin_vals.std()

    # Erode the brain label by 'd' mm.
    brain_label_itk = brain_label.astype('uint8')
    brain_label_itk = sitk.GetImageFromArray(brain_label_itk)
    brain_label_itk.SetSpacing(tuple(reversed(spacing)))
    brain_label_dist_map = sitk.SignedMaurerDistanceMap(brain_label_itk, useImageSpacing=True, squaredDistance=False, insideIsPositive=False)
    brain_label_dist_map = sitk.GetArrayFromImage(brain_label_dist_map)
    brain_label[brain_label_dist_map > -d] = 0
    if brain_label.sum() == 0:
        raise ValueError(f"Eroded brain label is empty, choose a larger region (?).")

    # Calculate "background noise" in the brain.
    hu_brain_eroded = image[np.nonzero(brain_label)]
    background_noise = hu_brain_eroded.std()
    
    # Calculate signal-to-noise ratio.
    snr = np.abs(struct_mean - margin_mean) / background_noise

    return snr

def mean_intensity(
    image: np.ndarray,
    label: np.ndarray) -> float:
    if image.shape != label.shape:
        raise ValueError(f"Metric 'mean_intensity' expects images of equal shape. Got '{image.shape}' and '{label.shape}'.")
    if (image.dtype != np.float32 and image.dtype != np.float64) or label.dtype != np.bool_:
        raise ValueError(f"Metric 'mean_intensity' expects (float, boolean) images. Got '{image.dtype}' and '{label.dtype}'.")
    if label.sum() == 0:
        raise ValueError(f"Metric 'mean_intensity' can't be calculated with an empty 'label'.")

    mean_intensity = image[np.nonzero(label)].mean()

    return mean_intensity
