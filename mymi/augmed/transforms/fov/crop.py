from typing import *

from mymi.typing import *

from ...utils import *
from ..random import RandomTransform

# The API should allow complexity but reduce to simplicity.
# E.g. rotation=5 -> rotation=(-5, 5, -5, 5) - for 2D - for each axis is a simple
# kwarg to pass, however we can specify non-symmetric rotations
# if required e.g. rotation=(-5, 10) -> rotation=(-5, 10, -5, 10) for 2D.

# crop_margin:
# - cm=(80, 120) -> cm=(80, 120, 80, 120, 80, 120, 80, 120, 80, 120, 80, 120) for 3D.
class RandomCrop(RandomTransform):
    def __init__(
        self,
        crop_margin: Optional[Union[Number, Tuple[Number, ...]]] = None,
        # Must keep 'centre' and 'centre_range' separate so we can specify image centre using 'centre'.
        centre: Union[Literal['centre'], Tuple[Union['centre', Number], ...], PointArray, PointTensor] = 'centre',
        centre_offset: Union[Number, Tuple[Number, ...]] = 20.0,
        **kwargs) -> None:
        super().__init__(**kwargs)
        crop_margin_range = expand_range_arg(crop_margin, negate_lower=False, vals_per_dim=4)
        assert len(crop_margin_range) == 4 * self._dim, f"Expected 'crop_margin' of length {4 * self._dim}, got {len(crop_margin_range)}."
        self.__crop_margin_range = to_tensor(crop_margin_range).reshape(self._dim, 2, 2)
        centre = arg_to_list(centre, (int, float, str), broadcast=self._dim)
        assert len(centre) == self._dim, f"Expected 'centre' of length {self._dim}, got {len(centre)}."
        self.__centre = centre  # Can't be tensor as might have 'centre' str.
        centre_offset_range = expand_range_arg(centre_offset, negate_lower=True, vals_per_dim=2)
        assert len(centre_offset_range) == 2 * self._dim, f"Expected 'centre_offset' of length {2 * self._dim}, got {len(centre_offset_range)}."
        self.__centre_offset_range = to_tensor(centre_offset_range).reshape(self._dim, 2)
        self._params = dict(
            centre=self.__centre,
            centre_offset_range=self.__centre_offset_range,
            crop_margin_range=self.__crop_margin_range,
            dim=self._dim,
            p=self._p,
        )

    def freeze(self) -> 'Crop':
        should_apply = self._rng.random(1) < self._p
        if not should_apply:
            return Identity(dim=self._dim)
        draw = to_tensor(self._rng.random((self._dim, 2)))
        crop_margin_draw = draw * (self.__crop_margin_range[:, :, 1] - self.__crop_margin_range[:, :, 0]) + self.__crop_margin_range[:, :, 0]
        draw = to_tensor(self._rng.random(self._dim))
        centre_offset_draw = draw * (self.__centre_offset_range[:, 1] - self.__centre_offset_range[:, 0]) + self.__centre_offset_range[:, 0]
        return Crop(crop_margin=crop_margin_draw, centre=self.__centre, centre_offset=centre_offset_draw, dim=self._dim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__crop_margin_range.flatten())}, centre={to_tuple(self.__centre)}, centre_offset={to_tuple(self.__centre_offset_range.flatten())}, dim={self._dim}, p={self._p})"

# centre:
# - c='centre' -> image centre used for translation.
# - cm=100 -> cm=(100, 100, 100, 100, 100, 100) for 3D, cm=(100, 100, 100, 100) for 2D.
# - cm=(50, 100) -> cm=(50, 100, 50, 100, 50, 100) for 3D, cm=(50, 100, 50, 100) for 2D.
class Crop(FovTransform):
    def __init__(
        self,
        crop_margin: Optional[Union[Number, Tuple[Number, ...]]] = None,
        # Must keep 'centre' and 'centre_range' separate so we can specify image centre using 'centre'.
        centre: Union[Literal['centre'], Tuple[Union['centre', Number], ...], PointArray, PointTensor] = 'centre',
        centre_offset: Union[Number, Tuple[Number, ...], PointArray, PointTensor] = 0.0,
        crop_trim: Optional[Union[Number, Tuple[Number, ...]]] = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.__centre = arg_to_list(centre, (int, float, str), broadcast=self._dim)
        assert crop_margin is not None or crop_trim is not None
        self.__crop_margin = expand_range_arg(crop_margin, dim=self._dim, negate_lower=False)
        self._params = dict(
            dim=self._dim,
            centre=self.__centre,
            centre_offset=self.__centre_offset,
            crop_margin=self.__crop_margin,
        )
        
        # Crop margin should show the crop on either side.
        # What params does crop have?
        # - centre: from where is the crop margin applied? Only required
        # if using crop_margin, deefaults to imag.
        # - crop_margin: how much of the image is left on either side of
        # the centre point.
        # - crop_trim: how much of the image is removed at each edge -
        # doesn't require a centre point.
        # - label: can be used to determine the crop centre and margin.
        # This is an interesting one as a crop transform occurring in the 
        # middle of a pipeline will reference a label that has already been
        # transformed. We could either perform the intermediate transform
        # for the label (using back_transform_points, transform_image) and
        # find it's location, or we could perform transform_points for the
        # boundary points. We have to do it for all boundary points, not
        # just the extrema, because elastic transforms could change extrema.
