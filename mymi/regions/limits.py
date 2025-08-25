import numpy as np

from mymi.geometry import fov_width
from mymi import logging
from mymi.transforms import crop_foreground
from mymi import typing

# Limits in mm.
class RegionLimits:
    SpinalCord = (-1, -1, 290)  # Assuming first-stage has spacing (4, 4, 4).

def truncate_spine(
    pred: np.ndarray,
    spacing: typing.Spacing3D) -> np.ndarray:
    ext_width = fov_width(pred, spacing)
    if ext_width is not None and ext_width[2] > RegionLimits.SpinalCord[2]:
        # Crop caudal end of spine.
        logging.info(f"Truncating caudal end of 'SpinalCord'. Got length (z-axis) of '{ext_width[2]}mm', maximum is '{RegionLimits.SpinalCord[2]}mm'.")
        top_z = extent(pred)[1][2]
        bottom_z = int(np.ceil(top_z - RegionLimits.SpinalCord[2] / spacing[2]))
        crop = ((0, 0, bottom_z), tuple(np.array(pred.shape) - 1))
        pred = crop_foreground(pred, crop, use_patient_coords=False)

    return pred
