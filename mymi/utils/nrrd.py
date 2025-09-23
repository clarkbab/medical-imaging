import nrrd

from mymi.typing import *

def load_nrrd(filepath: str) -> Tuple[ImageArray, Spacing3D, Point3D]:
    data, header = nrrd.read(filepath)
    affine = header['space directions']
    assert affine.sum() == np.diag(affine).sum()
    spacing = tuple(np.array([abs(affine[0][0]), abs(affine[1][1]), abs(affine[2][2])]).tolist())
    origin = tuple(np.array(header['space origin']).tolist())
    return data, spacing, origin
