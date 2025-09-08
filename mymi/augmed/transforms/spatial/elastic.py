from typing import *

from mymi.geometry import fov_centre, fov_width
from mymi.typing import *
from mymi.utils import *

from ...utils import *
from ..mixins import TransformImageMixin, TransformMixin
from ..random import RandomTransform
from .identity import IdentityTransform
from .spatial import SpatialTransform

# control spacing:
# - c=5 -> c=(5, 5, 5, 5) for 2D, c=(5, 5, 5, 5, 5, 5) for 3D. I.e deterministic control spacing.
# - c=(3, 5) -> c=(3, 5, 3, 5) for 2D, c=(3, 5, 3, 5, 3, 5) for 3D.
# disp:
# - d=5 -> d=(-5, 5, -5, 5) for 2D, d=(-5, 5, -5, 5, -5, 5) for 3D.
# - d=(3, 5) -> d=(3, 5, 3, 5) for 2D, d=(3, 5, 3, 5, 3, 5) for 3D.
# control origin:
# - c=5 -> c=(-5, 5, -5, 5) for 2D, c=(-5, 5, -5, 5, -5, 5) for 3D.
# - c=(3, 5) -> c=(3, 5, 3, 5) for 2D, c=(3, 5, 3, 5, 3, 5) for 3D.
class RandomElastic(RandomTransform):
    def __init__(
        self, 
        # Note that image folding may occur if the displacements are more than half the magnitude
        # of the control grid spacings. Add some sort of warning.
        control_spacing: Union[Number, Tuple[Number], np.ndarray, torch.Tensor] = 100.0,   # In mm.
        disp: Union[Number, Tuple[Number], np.ndarray, torch.Tensor] = 40.0,    # In mm.
        control_origin: Union[Number, Tuple[Number], np.ndarray, torch.Tensor] = 50.0,  # In mm.
        # Would other curve-fitting methods have desirable properties for certain situations?
        # E.g. 'bezier' allows sharp discontinuities which could be good for shearing in lung.
        # Can we randomise the curve-fitting method to increase the space of data augmentation?
        # This is the method for interpolating between points on the coarse control grid.
        method: Literal['bspline', 'cubic', 'linear'] = 'linear',
        dim: int = 3,
        p: float = 1.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert dim in [2, 3], "Only 2D and 3D elastic deformations are supported."
        self._dim = dim
        self.__method = method
        if isinstance(control_spacing, (int, float)):
            control_spacing_ranges = (control_spacing, control_spacing) * self._dim
        elif len(control_spacing) == 2:
            control_spacing_ranges = control_spacing * self._dim
        else:
            control_spacing_ranges = control_spacing
        assert len(control_spacing_ranges) == 2 * self._dim, f"Expected 'control' of length {2 * self._dim}, got {len(control_spacing_ranges)}."
        self.__control_spacing_ranges = to_tensor(control_spacing_ranges).reshape(self._dim, 2)
        if isinstance(control_origin, (int, float)):
            control_origin_ranges = (-control_origin, control_origin) * self._dim
        elif len(control_origin) == 2:
            control_origin_ranges = control_origin * self._dim
        else:
            control_origin_ranges = control_origin
        assert len(control_origin_ranges) == 2 * self._dim, f"Expected 'control_origin' of length {2 * self._dim}, got {len(control_origin_ranges)}."
        self.__control_origin_ranges = to_tensor(control_origin_ranges).reshape(self._dim, 2)
        if isinstance(disp, (int, float)):
            disp_ranges = (-disp, disp) * self._dim
        elif len(disp) == 2:
            disp_ranges = disp * self._dim
        else:
            disp_ranges = disp
        assert len(disp_ranges) == 2 * self._dim, f"Expected 'disp' of length {2 * self._dim}, got {len(disp_ranges)}."
        self.__disp_ranges = to_tensor(disp_ranges).reshape(self._dim, 2)
        self.__p = p
        self._params = dict(
            control_origin_ranges=self.__control_origin_ranges,
            control_spacing_ranges=self.__control_spacing_ranges,
            dim=self._dim,
            disp_ranges=self.__disp_ranges,
            method=self.__method,
            p=self.__p,
        )

    def freeze(self) -> 'Elastic':
        should_apply = self._rng.random(1) < self.__p
        if not should_apply:
            return IdentityTransform()
        draw = to_tensor(self._rng.random(self._dim))
        control_spacing_draw = draw * (self.__control_spacing_ranges[:, 1] - self.__control_spacing_ranges[:, 0]) + self.__control_spacing_ranges[:, 0]
        control_origin_draw = draw * (self.__control_origin_ranges[:, 1] - self.__control_origin_ranges[:, 0]) + self.__control_origin_ranges[:, 0]
        # We can't draw displacements here as we need the image to determine the number of control points.
        # However, we should pass a randomly-drawn seed.
        draw_rs = self._rng.integers(1e9)   # Requires upper bound.
        return Elastic(control_spacing=control_spacing_draw, disp=self.__disp_ranges, control_origin=control_origin_draw, dim=self._dim, method=self.__method, random_seed=draw_rs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__control_spacing_ranges.flatten())}, {to_tuple(self.__disp_ranges.flatten())}, {to_tuple(self.__control_origin_ranges.flatten())}, dim={self._dim}, p={self.__p})"

# Defines a coarse grid of control points.
# Random displacements are assigned at each control point.
# A b-spline is fitted (input: control point locations, output: perturbed control point positions)
# and this b-spline is used as the back transform.
# Cubic b-splines require a min of 4 control points per axis.
# Define max displacement, with some reasonable default.
# An option to set border displacements to zero. Maybe?

# control spacing:
# - c=5 -> c=(5, 5) for 2D, c=(5, 5, 5) for 3D.
# disp:
# - d=5 -> d=(-5, 5, -5, 5) for 2D, d=(-5, 5, -5, 5, -5, 5) for 3D.
# - d=(3, 5) -> d=(3, 5, 3, 5) for 2D, d=(3, 5, 3, 5, 3, 5) for 3D.
# control origin:
# - c=5 -> c=(5, 5) for 2D, c=(5, 5, 5) for 3D.
class Elastic(TransformImageMixin, TransformMixin, SpatialTransform):
    def __init__(
        self,
        random_seed: int,
        control_spacing: Union[Number, Spacing, SpacingArray, SpacingTensor] = 100.0,
        disp: Union[Number, Tuple[Number], np.ndarray, torch.Tensor] = 40.0,
        control_origin: Union[Number, Point, PointArray, PointTensor] = 20.0,
        dim: int = 3,
        method: Literal['bspline', 'cubic', 'linear'] = 'linear') -> None:
        assert dim in [2, 3], "Only 2D and 3D elastic deformations are supported."
        self._dim = dim
        assert method in ['bspline', 'cubic', 'linear'], "Only 'bspline', 'cubic', and 'linear' elastic methods are supported."
        self.__method = method
        self._is_homogeneous = False
        control_spacing = arg_to_list(control_spacing, (int, float), broadcast=dim)
        assert len(control_spacing) == dim, f"Expected 'control_spacing' of length '{dim}' for dim={dim}, got {len(control_spacing)}."
        self.__control_spacing = to_tensor(control_spacing)
        control_origin = arg_to_list(control_origin, (int, float), broadcast=dim)
        assert len(control_origin) == dim, f"Expected 'control_origin' of length '{dim}' for dim={dim}, got {len(control_origin)}."
        self.__control_origin = to_tensor(control_origin)
        # Disps aren't known until presented with the image.
        if isinstance(disp, (int, float)):
            disp_ranges = (-disp, disp) * self._dim
        elif len(disp) == 2:
            disp_ranges = disp * self._dim
        else:
            disp_ranges = disp
        self.__disp_ranges = disp_ranges
        self.__random_seed = random_seed
        self._params = dict(
            control_origin=self.__control_origin,
            control_spacing=self.__control_spacing,
            dim=self._dim,
            disp_ranges=self.__disp_ranges,
            random_seed=self.__random_seed,
        )

    # Control grid returned will change depending upon method.
    # E.g. cubic needs two control points outside whilst linear requires only one.
    def __get_control_grid(
        self,
        point_min: PointTensor,
        point_max: PointTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get CP extrema for this point cloud.
        cp_spacing = self.__control_spacing.to(point_min.device)
        cp_origin = self.__control_origin.to(point_min.device)
        cp_idx_min = torch.floor((point_min - cp_origin) / cp_spacing)
        cp_idx_max = torch.ceil((point_max - cp_origin) / cp_spacing)

        # Create grid of control points.
        cps = torch.stack(torch.meshgrid([
            torch.arange(cp_idx_min[a], cp_idx_max[a] + 1) for a in range(self._dim)
        ], indexing='ij'), dim=-1)
        cps = to_tensor(cps, device=point_min.device)
        cps = cps * cp_spacing + cp_origin

        # Seed the random number generator for deterministic transforms.
        # If the same 'points' are passed to 'back_transform_points' then the
        # transform is deterministic. Actually the same min/max points as these
        # determine the control grid size.
        self.__rng = np.random.default_rng(seed=self.__random_seed)

        # Draw random displacement at each control point.
        draw = to_tensor(self.__rng.random(cps.shape), device=point_min.device)
        disp_ranges = self.__disp_ranges.to(point_min.device)
        cp_disps = draw * (disp_ranges[:, 1] - disp_ranges[:, 0]) + disp_ranges[:, 0]

        return cps, cp_disps

    def back_transform_points(
        self,
        *args,
        **kwargs) -> PointsTensor:
        if self.__method == 'linear':
            return self.__back_transform_points_linear(*args, **kwargs)
        elif self.__method == 'cubic':
            return self.__back_transform_points_cubic(*args, **kwargs)
        elif self.__method == 'bspline':
            return self.__back_transform_points_bspline(*args, **kwargs)
        else:
            raise ValueError(f"Unrecognised elastic method '{self.__method}'.")

    def __back_transform_points_linear(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeArray, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor]] = None,
        **kwargs) -> PointsTensor:
        print('back transform')
        print(type(points))
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'

        print('performing elastic back transform points')

        # Get control grid.
        p_min, _ = points.min(dim=0)
        p_max, _ = points.max(dim=0)
        cps, cp_disps = self.__get_control_grid(p_min, p_max)

        # Create a new control grid origin for this point cloud.
        # Makes interpolation easier.
        cp_spacing = self.__control_spacing.to(points.device)
        cp_origin = self.__control_origin.to(points.device)
        cp_pc_idx_min = torch.floor((p_min - cp_origin) / cp_spacing)
        cp_pc_idx_max = torch.ceil((p_max - cp_origin) / cp_spacing)
        cp_pc_origin = cp_pc_idx_min * cp_spacing + cp_origin

        # Normalise points to the control grid.
        points_norm = (points - cp_pc_origin) / cp_spacing

        # Get lowest corner point.
        corner_min = torch.stack([torch.searchsorted(torch.arange(cps.shape[a]).to(points.device), points_norm[:, a]) - 1 for a in range(self._dim)], dim=-1)

        # Get distances from corner.
        u = points_norm - corner_min
        b = torch.stack([1 - u, u], dim=-2)

        # Get corner point offsets.
        offsets = torch.stack(torch.meshgrid([torch.tensor([0, 1]) for _ in range(self._dim)], indexing='ij'), dim=-1)
        offsets = offsets.reshape(-1, self._dim).to(points.device)

        # Calculate corners for each point.
        corners = corner_min[:, None, :] + offsets[None, :, :]

        # Split into x/y/z indices to perform control point disp selection.
        idxs = corners.unbind(-1)
        print('idxs: ', idxs[0].device)
        print('cp_disps: ', cp_disps.device)
        corner_disps = cp_disps[*idxs]

        # Create V of corner point displacements.
        V = corner_disps.reshape(-1, *(2, ) * self._dim, self._dim)

        # Compute interpolated displacements.
        if self._dim == 2:
            disps = torch.einsum('ni,nj,nijd->nd', b[:, :, 0], b[:, :, 1], V)
        elif self._dim == 3:
            disps = torch.einsum('ni,nj,nk,nijkd->nd', b[:, :, 0], b[:, :, 1], b[:, :, 2], V)

        # Get displaced input points.
        points_t = points + disps

        print('points')
        print(type(points_t))
        print(return_type)

        if return_type == 'numpy':
            points_t = to_array(points_t)
        return points_t

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__control_spacing)}, {to_tuple(self.__disp_ranges.flatten())}, {to_tuple(self.__control_origin)}, random_seed={self.__random_seed}, dim={self._dim})"
