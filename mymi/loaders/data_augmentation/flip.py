import numpy as np
import SimpleITK as sitk
from typing import *

from mymi.geometry import fov_centre
from mymi.typing import *
from mymi.utils import *

class RandomFlip:
    def __init__(
        self,
        p: Union[float, Tuple[float, float, float]] = 0.5,
        **kwargs) -> None:
        # Convert arguments.
        p = arg_to_list(p, float, broadcast=3)
        p = tuple(p)
        assert len(p) == 3, f"Expected p to be a float or a tuple of 3 floats, got {p}."
        self.__p = p
        logging.info(f"Using sitk flip augmentation: p={p}.")

    def get_concrete_transform(
        self,
        # Size/spacing/offset are required to set the centre [mm] of the flip transform.
        size: Size3D,
        spacing: Spacing3D,
        offset: Point3D,
        random_seed: Optional[int] = None,
        **kwargs) -> Tuple[sitk.Transform, sitk.Transform, Dict[str, Any]]:
        # Sample transform parameters.
        if random_seed is not None:
            np.random.seed(random_seed)
        p_draw = [np.random.choice([1, -1], p=[1 - p, p]) for p in self.__p]
        print('drawn p:', p_draw)
        print(size, spacing, offset)

        # Create transform.
        flip_transform = sitk.AffineTransform(3)
        matrix = np.diag(p_draw)
        flip_transform.SetMatrix(matrix.flatten().tolist())
        centre = fov_centre(np.zeros(size), spacing=spacing, offset=offset, use_patient_coords=True)
        print('centre:', centre)
        flip_transform.SetCenter(list(centre))

        # The inverse transform should be the same as the forward for flipping.
        forward_transform = flip_transform
        backward_transform = flip_transform

        # Return params for debugging.
        params = {
            'p': p_draw,
        }

        return forward_transform, backward_transform, params
