from typing import *

from mymi.typing import *

from ...utils import *
from ..identity import Identity
from .grid import GridTransform
from .random import RandomGridTransform

# The API should allow complexity but reduce to simplicity.
# E.g. rotation=5 -> rotation=(-5, 5, -5, 5) - for 2D - for each axis is a simple
# kwarg to pass, however we can specify non-symmetric rotations
# if required e.g. rotation=(-5, 10) -> rotation=(-5, 10, -5, 10) for 2D.

# pad:
# - c=(80, 120) -> c=(80, 120, 80, 120, 80, 120, 80, 120, 80, 120, 80, 120) for 3D.
# Means that for each end of each axis (6 ends), we can take between 80 and 120mm
# off.
# What if want symmetric random pads?
# We need to add a parameter for this.
# - c=100 -> c=(100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100) for 3D.
# This gives us a static pad amount of 100mm removed from each axis. Is this what we want?
# Probably, if they want to remove a random amount, use c=(0, 100).
# pad_margin:
# - cm=(80, 120) -> cm=(80, 120, 80, 120, 80, 120, 80, 120, 80, 120, 80, 120) for 3D.
class RandomPad(RandomGridTransform):
    def __init__(
        self,
        # How do we handle the case where we want a symmetric pad?
        # For example, we may define pad as: (50, 50, 50, 50, 50, 50, 50, 50,)
        pad_range: Optional[Union[Number, Tuple[Number, ...]]] = None,
        pad_margin_range: Optional[Union[Number, Tuple[Number, ...]]] = None,
        # Must keep 'centre' and 'centre_range' separate so we can specify image centre using 'centre'.
        centre: Union[Literal['centre'], Tuple[Union['centre', Number], ...], PointArray, PointTensor] = 'centre',
        centre_offset_range: Union[Number, Tuple[Number, ...]] = 0.0,
        # padped amounts are the same at both ends of each axis.
        # This should be configured per axis really, for example we might want want symmetry
        # along the x-axis only.
        symmetric: Union[bool, Tuple[bool, ...]] = False,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert pad_range is not None or pad_margin_range is not None
        self.__symmetric = to_tensor(symmetric, broadcast=self._dim)
        if pad_range is not None:
            # Handle pad from outside case.
            cr_vals_per_dim = 4
            pad_range = expand_range_arg(pad_range, dim=self._dim, vals_per_dim=cr_vals_per_dim)
            assert len(pad_range) == cr_vals_per_dim * self._dim, f"Expected 'pad_range' of length {cr_vals_per_dim * self._dim}, got {len(pad_range)}."

            # Ensure pad ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cr_axis_vals = pad_range[i * cr_vals_per_dim:(i + 1) * cr_vals_per_dim]
                if s and (cr_axis_vals[0] != cr_axis_vals[2] or cr_axis_vals[1] != cr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad ranges {cr_axis_vals}.")

            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__pad_range = to_tensor(pad_range, dtype=dtype).reshape(self._dim, 2, 2)
            # Should we zero out things that aren't relevant?
            self.__pad_margin_range = None
            self.__centre = None
            self.__centre_offset_range = None
        else:
            # Handle pad from centre point and margin case.
            self.__pad_range = None
            cmr_vals_per_dim = 4
            pad_margin_range = expand_range_arg(pad_margin_range, dim=self._dim, vals_per_dim=cmr_vals_per_dim)
            assert len(pad_margin_range) == cmr_vals_per_dim * self._dim, f"Expected 'pad_margin_range' of length {cmr_vals_per_dim * self._dim}, got {len(pad_margin_range)}."

            # Ensure pad margin ranges allow symmetry.
            for i, s in enumerate(self.__symmetric):
                cmr_axis_vals = pad_margin_range[i * cmr_vals_per_dim:(i + 1) * cmr_vals_per_dim]
                if s and (cmr_axis_vals[0] != cmr_axis_vals[2] or cmr_axis_vals[1] != cmr_axis_vals[3]):
                    raise ValueError(f"Cannot create symmetric pads for axis {i} with pad margin ranges {cmr_axis_vals}.")

            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__pad_margin_range = to_tensor(pad_margin_range, dtype=dtype).reshape(self._dim, 2, 2)
            centre = arg_to_list(centre, (int, float, str), broadcast=self._dim)
            assert len(centre) == self._dim, f"Expected 'centre' of length {self._dim}, got {len(centre)}."
            self.__centre = centre  # Can't be tensor as might have 'centre' str.
            centre_offset_range = expand_range_arg(centre_offset_range, dim=self._dim, negate_lower=True)
            assert len(centre_offset_range) == 2 * self._dim, f"Expected 'centre_offset_range' of length {2 * self._dim}, got {len(centre_offset_range)}."
            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__centre_offset_range = to_tensor(centre_offset_range, dtype=dtype).reshape(self._dim, 2)

        self._params = dict(
            centre=self.__centre,
            centre_offset_range=self.__centre_offset_range,
            dim=self._dim,
            pad_range=self.__pad_range,
            pad_margin_range=self.__pad_margin_range,
            p=self._p,
        )

    def freeze(self) -> 'Pad':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random((self._dim, 2)))
        if self.__pad_range is not None:
            dtype = torch.int32 if self._use_image_coords else torch.float32
            pad_draw = (draw * (self.__pad_range[:, :, 1] - self.__pad_range[:, :, 0]) + self.__pad_range[:, :, 0]).type(dtype)
            # Copy lower end of axis for symmetric pads.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            pad_draw[sym_axes, 1] = pad_draw[sym_axes, 0]
            pad_margin_draw = None
            centre_offset_draw = None
        else:
            pad_draw = None
            dtype = torch.int32 if self._use_image_coords else torch.float32
            pad_margin_draw = (draw * (self.__pad_margin_range[:, :, 1] - self.__pad_margin_range[:, :, 0]) + self.__pad_margin_range[:, :, 0]).type(dtype)
            # Copy lower end of axis for symmetric pads.
            sym_axes = torch.argwhere(self.__symmetric).flatten()
            pad_margin_draw[sym_axes, 1] = pad_margin_draw[sym_axes, 0]
            draw = to_tensor(self._rng.random(self._dim))
            dtype = torch.int32 if self._use_image_coords else torch.float32
            centre_offset_draw = (draw * (self.__centre_offset_range[:, 1] - self.__centre_offset_range[:, 0]) + self.__centre_offset_range[:, 0]).type(dtype)

        params = dict(
            pad=pad_draw,
            pad_margin=pad_margin_draw,
            centre=self.__centre,
            centre_offset=centre_offset_draw,
        )
        return super().freeze(Pad, params)

    def __str__(self) -> str:
        params = dict(
            pad=to_tuple(self.__pad_range.flatten()) if self.__pad_range is not None else None,
            pad_margin=to_tuple(self.__pad_margin_range.flatten()) if self.__pad_margin_range is not None else None,
            centre=to_tuple(self.__centre),
            centre_offset=to_tuple(self.__centre_offset.flatten()) if self.__centre_offset is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

# centre:
# - c='centre' -> image centre used for translation.
# - cm=100 -> cm=(100, 100, 100, 100, 100, 100) for 3D, cm=(100, 100, 100, 100) for 2D.
# - cm=(50, 100) -> cm=(50, 100, 50, 100, 50, 100) for 3D, cm=(50, 100, 50, 100) for 2D.
class Pad(GridTransform):
    def __init__(
        self,
        pad: Optional[Union[Number, Tuple[Number, ...]]] = None,
        pad_margin: Optional[Union[Number, Tuple[Number, ...]]] = None,
        # Must keep 'centre' and 'centre_range' separate so we can specify image centre using 'centre'.
        centre: Union[Literal['centre'], Tuple[Union['centre', Number], ...], PointArray, PointTensor] = 'centre',
        centre_offset: Union[Number, Tuple[Number, ...], PointArray, PointTensor] = 0.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert pad is not None or pad_margin is not None
        if pad is not None:
            pad = expand_range_arg(pad, dim=self._dim)
            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__pad = to_tensor(pad, dtype=dtype).reshape(self._dim, 2)
            self.__pad_margin = None
            self.__centre = None
            self.__centre_offset = None
        else:
            pad_margin = expand_range_arg(pad_margin, dim=self._dim)
            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__pad_margin = to_tensor(pad_margin, dtype=dtype).reshape(self._dim, 2)
            self.__centre = to_tuple(centre, broadcast=self._dim)   # Tensors can't store str types.
            assert len(self.__centre) == self._dim
            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__centre_offset = to_tensor(centre_offset, broadcast=self._dim, dtype=dtype)
            assert len(self.__centre_offset) == self._dim

        self._params = dict(
            dim=self._dim,
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            pad=self.__pad,
            pad_margin=self.__pad_margin,
        )

    def __str__(self) -> str:
        params = dict(
            pad=to_tuple(self.__pad.flatten()) if self.__pad is not None else None,
            pad_margin=to_tuple(self.__pad_margin.flatten()) if self.__pad_margin is not None else None,
            centre=to_tuple(self.__centre),
            centre_offset=to_tuple(self.__centre_offset.flatten()) if self.__centre_offset is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_grid(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        if self.__pad is not None:
            # Get pad box.
            pad_min_mm = self.__pad[:, 0].to(size.device)
            pad_max_mm = self.__pad[:, 1].to(size.device)
            if self._use_image_coords:
                # Convert from image -> patient coords.
                pad_min_mm = (spacing * pad_min_mm).type(torch.float32)
                pad_max_mm = (spacing * pad_max_mm).type(torch.float32)
            pad_min_mm = origin - pad_min_mm
            pad_max_mm = origin + size * spacing + pad_max_mm

            # Convert to voxels.
            pad_min_vox = torch.round((pad_min_mm - origin) / spacing).type(torch.int32)
            pad_max_vox = torch.round((pad_max_mm - origin) / spacing).type(torch.int32)
        else:
            # Get pad centre.
            centre_mm = to_tensor([oi + (si * spi) / 2 if c == 'centre' else c for c, si, spi, oi in zip(self.__centre, size, spacing, origin)], device=size.device)
            centre_mm = centre_mm + self.__centre_offset.to(size.device)

            # Get pad box.
            pad_margin_min_mm = self.__pad_margin[:, 0].to(size.device)
            pad_margin_max_mm = self.__pad_margin[:, 1].to(size.device)
            if self._use_image_coords:
                # Convert from image -> patient coords.
                pad_margin_min_mm = (spacing * pad_margin_min_mm).type(torch.float32)
                pad_margin_max_mm = (spacing * pad_margin_max_mm).type(torch.float32)
            pad_min_mm = centre_mm - pad_margin_min_mm
            pad_max_mm = centre_mm + pad_margin_max_mm

            # Convert to voxels.
            pad_min_vox = torch.round((pad_min_mm - origin) / spacing).type(torch.int32)
            pad_max_vox = torch.round((pad_max_mm - origin) / spacing).type(torch.int32)

            # Truncate to true voxel coords.
            pad_min_vox = torch.clamp(pad_min_vox, 0)
            pad_max_vox = torch.clamp(pad_max_vox, max=(size - 1))

        # Get new FOV.
        size_t = pad_max_vox - pad_min_vox
        origin_t = (pad_min_vox * spacing) + origin

        return size_t, spacing, origin_t

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        filter_offgrid: bool = True,
        return_filtered: bool = False,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        origin = to_tensor(origin, device=points.device, dtype=points.dtype)
        size = to_tensor(size, device=points.device, dtype=points.dtype)
        spacing = to_tensor(spacing, device=points.device, dtype=points.dtype)

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        if filter_offgrid:
            assert origin is not None
            assert size is not None
            assert spacing is not None
            # Get new FOV.
            size_t, spacing_t, origin_t = self.transform_grid(sz, sp, o)

            # Get pad box.
            pad_min_mm = origin_t
            pad_max_mm = origin_t + size_t * spacing_t

            # Pad points.
            pad_mm = torch.stack([pad_min_mm, pad_max_mm]).to(points.device)
            print(pad_mm)
            to_keep = (points >= pad_mm[0]) & (points < pad_mm[1])
            print(to_keep)
            to_keep = to_keep.all(axis=1)
            points_t = points[to_keep]
            indices = torch.where(to_keep)[0]
            if return_type == 'numpy':
                points_t, indices = points_t.numpy(), indices.numpy()
            if return_filtered:
                return points_t, indices
            else:
                return points_t
        else:
            if return_type == 'numpy':
                points = points.numpy()
            return points
