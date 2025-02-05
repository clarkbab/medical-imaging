import nrrd

from mymi.typing import *

def load_nrrd(filepath: str) -> Tuple[np.ndarray, ImageSpacing3D, PointMM3D]:
    data, header = nrrd.read(filepath)
    affine = header['space directions']
    assert affine.sum() == np.diag(affine).sum()
    spacing = (abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2]))
    offset = tuple(header['space origin'])
    return data, spacing, offset
