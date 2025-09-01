from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..mixins import TransformImageMixin, TransformMixin
from ..random import RandomTransform
from .identity import IdentityTransform
from .spatial import SpatialTransform

# Should we randomise over control points or just disps?
# Start with just disps.
class RandomElastic(RandomTransform):
    def __init__(
        self, 
        # What are sensible defaults for n_control and disp?
        n_control: Union[int, Tuple[int], np.ndarray, torch.Tensor] = 10,
        # Allow non-symmetric displacement ranges? Maybe later.
        disp: Union[float, Tuple[float], np.ndarray, torch.Tensor] = 20.0,    # In mm.
        dim: int = 3,
        p: float = 1.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert dim in [2, 3], "Only 2D and 3D rotations are supported."
        self._dim = dim
        n_controls = arg_to_list(n_control, int, broadcast=dim)
        assert len(n_controls) == dim, f"Expected 'n_control' of length '{dim}' for dim={dim}, got {len(n_controls)}."
        self.__n_controls = to_tensor(n_controls, dtype=torch.int)
        disps = arg_to_list(disp, float, broadcast=dim)
        assert len(disps) == dim, f"Expected 'disp' of length '{dim}' for dim={dim}, got {len(disps)}."
        self.__disps = to_tensor(disps)
        self.__p = p
        self._params = dict(
            dim=self._dim,
            disps=self.__disps,
            n_controls=self.__n_controls,
            p=self.__p,
        )

    # How do we freeze displacements without access to image properties?
    # I.e. two images could have different spacings, but we want to be able to apply
    # the same frozen transform to both of these images.
    # We probably have to define displacements relative to some image value,
    # E.g. a disp of 1.0 in the x direction is actually 1.0 mm, not 1.0 voxel.
    def freeze(self) -> 'Elastic':
        should_apply = self._rng.random(1) < self.__p
        if not should_apply:
            return IdentityTransform()
        draw = to_tensor(self._rng.random((*to_array(self.__n_controls), self._dim)))
        draw_mm = 2 * (draw - 0.5) * self.__disps
        return Elastic(n_control=self.__n_controls, disp=draw_mm, dim=self._dim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__n_controls)}, {to_tuple(self.__disps)}, dim={self._dim}, p={self.__p})"

# Defines a coarse grid of control points.
# Random displacements are assigned at each control point.
# A b-spline is fitted (input: control point locations, output: perturbed control point positions)
# and this b-spline is used as the back transform.
# Cubic b-splines require a min of 4 control points per axis.
# Define max displacement, with some reasonable default.
# An option to set border displacements to zero. Maybe?

class Elastic(TransformImageMixin, TransformMixin, SpatialTransform):
    def __init__(
        self,
        # Can't really provide a default for 'disp', could for 'n_control' but we want to
        # match random transform param ordering.
        n_control: Union[int, Tuple[int], np.ndarray, torch.Tensor],
        disp: Union[np.ndarray, torch.Tensor],
        dim: int = 3) -> None:
        self._dim = dim
        self._is_homogeneous = False
        n_controls = arg_to_list(n_control, int, broadcast=dim)
        assert len(n_controls) == dim, f"Expected 'n_control' of length '{dim}' for dim={dim}, got {len(n_controls)}."
        self.__n_controls = to_tensor(n_controls, dtype=torch.int)
        disp_shape = (*to_array(self.__n_controls), self._dim)
        assert disp.shape == disp_shape, f"Expected 'disp' of shape '{disp_shape}' for dim={dim}, got {disp.shape}."
        self.__disps = to_tensor(disp)
        self._params = dict(
            dim=self._dim,
            disps=self.__disps,
            n_controls=self.__n_controls,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__n_controls)}, {to_tuple(self.__disps.shape)} <disp size>, dim={self._dim})"
