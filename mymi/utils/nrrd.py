from augmed.typing import *
import nrrd

def load_nrrd(
    filepath: str,
    ) -> Tuple[Image3D, AffineMatrix3D]:
    data, header = nrrd.read(filepath)
    affine = np.zeros((4, 4), dtype=np.float32)
    affine[:3, :3] = header['space directions']
    affine[:3, 3] = header['space origin']
    affine[3, 3] = 1.0
    return data, affine
