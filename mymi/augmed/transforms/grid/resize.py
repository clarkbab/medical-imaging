from typing import *

from mymi.typing import *

from ...utils import *
from ..random import RandomTransform
from .grid import GridTransform

# If we change the spacing, size should change to preserve the field-of-view.
# If we change the size, spacing should change to preserve the field-of-view.
# If we change both size/spacing, field-of-view is screwed.

# size:
# - s=100 -> s=(100, 100, ...)
# - s=(80, 120) -> s=(80, 120, ...)
# spacing:
# - s=1.0 -> s=(1.0, 1.0, ...)
# - s=(0.8, 1.2) -> s=(0.8, 1.2, ...)
class RandomResize(RandomTransform):
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,     # Not affected by use_image_coords?
        spacing: Optional[Union[float, Tuple[float, ...]]] = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert size is not None or spacing is not None
        if size is not None:
            size = expand_range_arg(size, dim=self._dim)
            assert len(size) == 2 * self._dim, f"Expected 'size' of length {2 * self._dim} for dim={self._dim}, got length {len(size)}."
            self.__size = to_tensor(size, dtype=torch.int32).reshape(self._dim, 2)
        else:
            self.__size = None
        if spacing is not None:
            spacing = expand_range_arg(spacing, dim=self._dim)
            assert len(spacing) == 2 * self._dim, f"Expected 'spacing' of length {2 * self._dim} for dim={self._dim}, got length {len(spacing)}."
            dtype = torch.int32 if self._use_image_coords else torch.float32
            self.__spacing = to_tensor(spacing, dtype=dtype).reshape(self._dim, 2)
        else:
            self.__spacing = None

        self._params = dict(
            dim=self._dim,
            p=self._p,
            size=self.__size,
            spacing=self.__spacing,
        )

    def freeze(self) -> 'Resize':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        if self.__size is not None:
            draw = to_tensor(self._rng.random(self._dim))
            size_draw = (draw * (self.__size[:, 1] - self.__size[:, 0]) + self.__size[:, 0]).type(torch.int32)
        else:
            size_draw = None
        if self.__spacing is not None:
            draw = to_tensor(self._rng.random(self._dim))
            dtype = torch.int32 if self._use_image_coords else torch.float32
            spacing_draw = (draw * (self.__spacing[:, 1] - self.__spacing[:, 0]) + self.__spacing[:, 0]).type(dtype)
        else:
            spacing_draw = None

        params = dict(
            size=size_draw,
            spacing=spacing_draw,
        )
        return super().freeze(Resize, params)

    def __str__(self) -> str:
        params = dict(
            size=to_tuple(self.__size.flatten()) if self.__size is not None else None,
            spacing=to_tuple(self.__spacing.flatten()) if self.__spacing is not None else None,
        )
        return super().__str__(self.__class__.__name__, params)

class Resize(GridTransform):
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,     # Not affected by uic.
        spacing: Optional[Union[float, Tuple[float, ...]]] = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        assert size is not None or spacing is not None
        self.__size = to_tensor(size, broadcast=self._dim, dtype=torch.int32) if size is not None else None
        dtype = torch.int32 if self._use_image_coords else torch.float32
        self.__spacing = to_tensor(spacing, broadcast=self._dim, dtype=dtype) if spacing is not None else None
        self._params = dict(
            dim=self._dim,
            size=self.__size,
            spacing=self.__spacing,
        )

    def __str__(self) -> str:
        params = dict(
            size=to_tuple(self.__size),
            spacing=to_tuple(self.__spacing),
        )
        return super().__str__(self.__class__.__name__, params)

    def transform_grid(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        origin_t = origin
        if self.__size is not None:
            if self.__spacing is not None:
                # Set new FOV.
                size_t = self.__size.to(size.device)
                spacing_t = self.__spacing.to(size.device)
                if self._use_image_coords:
                    # Convert from voxels -> mm.
                    spacing_t = (spacing_t * spacing).type(torch.float32)
            else:
                # Change spacing to maintain current FOV.
                size_t = self.__size.to(size.device)
                fov_max = origin + size * spacing
                spacing_t = (fov_max - origin) / size_t
        elif self.__spacing is not None:
            # Change size to maintain current FOV.
            spacing_t = self.__spacing.to(size.device)
            if self._use_image_coords:
                # Convert from voxels -> mm.
                spacing_t = (spacing_t * spacing).type(torch.float32)
            fov_max = origin + size * spacing
            size_t = torch.round((fov_max - origin) / spacing_t).type(torch.int32)

        return size_t, spacing_t, origin_t
