from functools import reduce
import numpy as np
from typing import Any, Dict, List, Optional

from mymi.metrics import mean_intensity, snr
from mymi.geometry import get_extent, get_extent_width_mm
from mymi.postprocessing import largest_cc_3D
from mymi.typing import *

def get_ct_stats(
    ct: np.ndarray,
    spacing: ImageSpacing3D) -> List[Dict[str, Any]]:
    # Extract required stats.
    size = ct.shape
    fov = np.array(size) * spacing

    stats = []
    for axis in range(3):
        data = {
            'axis': axis,
            'size': size[axis],
            'spacing': spacing[axis],
            'fov': fov[axis]
        }
        stats.append(data)

    return stats

def get_region_stats(
    ct_data: CtImage,
    region_data: RegionLabel,
    spacing: ImageSpacing3D,
    offset: Point3D,
    brain_data: Optional[RegionLabel] = None) -> List[Dict[str, Any]]:

    stats = []

    # Add 'min/max' extent metrics.
    data = {}
    min_extent_vox = np.argwhere(region_data).min(axis=0)
    min_extent_mm = min_extent_vox * spacing + offset
    max_extent_vox = np.argwhere(region_data).max(axis=0)
    max_extent_mm = max_extent_vox * spacing + offset
    for axis, min, max in zip(('x', 'y', 'z'), min_extent_mm, max_extent_mm):
        data['metric'] = f'min-extent-mm-{axis}'
        data['value'] = min
        stats.append(data.copy())
        data['metric'] = f'max-extent-mm-{axis}'
        data['value'] = max
        stats.append(data.copy())

    # Add 'connected' metrics.
    data['metric'] = 'connected'
    lcc_region_data = largest_cc_3D(region_data)
    data['value'] = 1 if lcc_region_data.sum() == region_data.sum() else 0
    stats.append(data.copy())
    data['metric'] = 'connected-largest-p'
    data['value'] = lcc_region_data.sum() / region_data.sum()
    stats.append(data.copy())

    # Add intensity metrics.
    if brain_data is not None:
        data['metric'] = 'snr-brain'
        data['value'] = snr(ct_data, region_data, brain_data, spacing)
        stats.append(data.copy())
    data['metric'] = 'mean-intensity'
    data['value'] = mean_intensity(ct_data, region_data)
    stats.append(data.copy())

    # Add OAR extent.
    ext_width_mm = get_extent_width_mm(region_data, spacing)
    if ext_width_mm is None:
        ext_width_mm = (0, 0, 0)
    data['metric'] = 'extent-mm-x'
    data['value'] = ext_width_mm[0]
    stats.append(data.copy())
    data['metric'] = 'extent-mm-y'
    data['value'] = ext_width_mm[1]
    stats.append(data.copy())
    data['metric'] = 'extent-mm-z'
    data['value'] = ext_width_mm[2]
    stats.append(data.copy())

    # Add extent of largest connected component.
    extent = get_extent(lcc_region_data)
    if extent:
        min, max = extent
        extent_vox = np.array(max) - min
        extent_mm = extent_vox * spacing
    else:
        extent_mm = (0, 0, 0)
    data['metric'] = 'connected-extent-mm-x'
    data['value'] = extent_mm[0]
    stats.append(data.copy())
    data['metric'] = 'connected-extent-mm-y'
    data['value'] = extent_mm[1]
    stats.append(data.copy())
    data['metric'] = 'connected-extent-mm-z'
    data['value'] = extent_mm[2]
    stats.append(data.copy())

    # Add volume.
    vox_volume = reduce(np.multiply, spacing)
    data['metric'] = 'volume-mm3'
    data['value'] = vox_volume * region_data.sum() 
    stats.append(data.copy())

    return stats
