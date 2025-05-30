import numpy as np

from mymi import typing

def get_box(
    centre: typing.Voxel,
    size: typing.Size3D) -> typing.VoxelBox:
    # Convert to box.
    size = np.array(size)
    lower_sub = np.ceil(size / 2).astype(int)
    min = tuple(centre - lower_sub)
    max = tuple(min + size)
    box = (min, max)

    return box
