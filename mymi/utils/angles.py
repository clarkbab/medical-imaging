import numpy as np
from typing import List, Literal

from .args import arg_to_list

def convert_angles(
    angles: List[float],
    from_: Literal['kv-detector', 'kv-source', 'mv-detector', 'mv-source'],
    to: Literal['kv-detector', 'kv-source', 'mv-detector', 'mv-source'],
    machine: Literal['elekta', 'varian'],
    scale: Literal['degrees', 'radians'] = 'degrees',
    ) -> List[float]:
    angle_360 = 2 * np.pi if scale == 'radians' else 360
    angle_180 = angle_360 / 2
    angle_90 = angle_360 / 4

    # Get offset from 'from_' to MVSource.
    if from_ == 'kv-detector':
        mv_offset = angle_90 if machine == 'elekta' else -angle_90
    elif from_ == 'kv-source':
        mv_offset = angle_90 if machine == 'varian' else -angle_90
    elif from_ == 'mv-detector':
        mv_offset = 180
    elif from_ == 'mv-source':
        mv_offset = 0
    else:
        raise ValueError(f"Unrecognised position 'from_={from_}'")

    # Get offset from MVSource to 'to'.
    if to == 'kv-detector':
        to_offset = angle_90 if machine == 'varian' else -angle_90
    elif to == 'kv-source':
        to_offset = angle_90 if machine == 'elekta' else -angle_90
    elif to == 'mv-detector':
        to_offset = angle_180
    elif to == 'mv-source':
        to_offset = 0
    else:
        raise ValueError(f"Unrecognised position 'to={to}'")

    # Convert angles.
    offset = mv_offset + to_offset
    angles = [float(np.round((a + offset) % angle_360, decimals=3)) for a in angles]
    return angles

def reverse_angles(
    angles: float | List[float],
    ) -> float | List[float]:
    angles, was_single = arg_to_list(angles, (int, float), return_matched=True)
    angles = [float(np.round((360 - a) % 360, decimals=3)) for a in angles]
    if was_single:
        return angles[0]
    else:
        return angles
