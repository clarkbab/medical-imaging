from typing import *

from mymi.typing import *

from ...utils import *
from ..random import RandomTransform
from .fov import FovTransform

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
        centre_offset: Union[Number, Tuple[Number, ...]] = 0.0,
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
        assert crop_margin is not None or crop_trim is not None
        crop_margin = expand_range_arg(crop_margin, dim=self._dim, negate_lower=False)
        self.__crop_margin = to_tensor(crop_margin).reshape(self._dim, 2)
        self.__centre = to_tuple(centre, broadcast=self._dim)   # Tensors can't store str types.
        assert len(self.__centre) == self._dim
        self.__centre_offset = to_tensor(centre_offset, broadcast=self._dim)
        assert len(self.__centre_offset) == self._dim
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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({to_tuple(self.__crop_margin.flatten())}) centre={to_tuple(self.__centre)}, centre_offset={to_tuple(self.__centre_offset)}, dim={self._dim})"

    def transform_fov(
        self,
        size: SizeTensor,
        spacing: SpacingTensor,
        origin: PointTensor,
        **kwargs) -> Tuple[SizeTensor, SpacingTensor, PointTensor]:
        # Get crop centre.
        centre_mm = to_tensor([oi + (si * spi) / 2 if c == 'centre' else c for c, si, spi, oi in zip(self.__centre, size, spacing, origin)], device=size.device)
        centre_mm = centre_mm + self.__centre_offset.to(size.device)

        # Get crop box.
        crop_min_mm = centre_mm - self.__crop_margin[:, 0].to(size.device)
        crop_max_mm = centre_mm + self.__crop_margin[:, 0].to(size.device)

        # Convert to voxels.
        crop_min_vox = torch.floor((crop_min_mm - origin) / spacing).type(torch.int)
        crop_max_vox = torch.ceil((crop_max_mm - origin) / spacing).type(torch.int)

        # Truncate to true voxel coords.
        crop_min_vox = torch.clamp(crop_min_vox, 0)
        crop_max_vox = torch.clamp(crop_max_vox, max=(size - 1))

        # Get new FOV.
        size = crop_max_vox - crop_min_vox
        origin = (crop_min_vox * spacing) + origin

        return size, spacing, origin

    # Just removes voxels outside the crop region.
    @alias_kwargs([
        ('o', 'origin'),
        ('rf', 'return_fov'),
        ('s', 'spacing'),
    ])
    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor, LabelArray, LabelTensor, List[Union[ImageArray, ImageTensor, LabelArray, LabelTensor]]],
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        return_fov: bool = False) -> Union[ImageArrayWithFov, ImageTensorWithFov, List[Union[ImageArrayWithFov, ImageTensorWithFov]]]:
        images, image_was_single = arg_to_list(image, (np.ndarray, torch.Tensor), return_matched=True)
        return_types = ['numpy' if isinstance(i, np.ndarray) else 'torch' for i in images]
        origins = arg_to_list(origin, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(images))
        spacings = arg_to_list(spacing, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(images))
        images = [to_tensor(i) for i in images]
        devices = [i.device for i in images]
        dims = [len(i.shape) for i in images]
        if self._dim == 2:
            for i, d in enumerate(dims):
                assert d in [2, 3, 4], f"Expected 2-4D image (2D spatial, optional batch/channel), got {d}D for image {i}."
        elif self._dim == 3:
            for i, d in enumerate(dims):
                assert d in [3, 4, 5], f"Expected 3-5D image (3D spatial, optional batch/channel), got {d}D for image {i}."
        sizes = [to_tensor(i.shape[-self._dim:], device=i.device, dtype=torch.int) for i in images]
        spacings = [to_tensor((1,) * self._dim, device=i.device) if s is None else to_tensor(s, device=i.device) for s, i in zip(spacings, images)]
        origins = [to_tensor((0,) * self._dim, device=i.device) if o is None else to_tensor(o, device=i.device) for o, i in zip(origins, images)]

        # Crop images.
        image_ts = []
        for i, (image, dim, sz, sp, o, dev, rt) in enumerate(zip(images, dims, sizes, spacings, origins, devices, return_types)):
            # Get new FOV.
            size_t, spacing_t, origin_t = self.transform_fov(sz, sp, o)

            # Convert to voxel crop.
            crop_min_vox = ((origin_t - o) / sp).type(torch.int)
            crop_max_vox = crop_min_vox + size_t

            # Perform crop.
            image_t = image[list(slice(a_min, a_max) for a_min, a_max in zip(crop_min_vox, crop_max_vox))]

            # Convert to return types.
            if rt == 'numpy': 
                image_t = to_array(image_t)
            if return_fov:
                fov_t = (size_t, spacing_t, origin_t)
                if rt == 'numpy':
                    fov_t = (to_array(f) for f in fov_t)
                image_ts.append((image_t, fov_t))
            else:
                image_ts.append(image_t)

        if image_was_single:
            return image_ts[0]
        else:
            return image_ts

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        size: Optional[Union[Size, SizeTensor]] = None,
        spacing: Optional[Union[Spacing, SpacingTensor]] = None,
        origin: Optional[Union[Point, PointTensor]] = None,
        filter_offscreen: bool = True,
        return_filtered: bool = False,
        **kwargs) -> Union[PointsArray, PointsTensor, Tuple[Union[PointsArray, PointsTensor], Union[np.ndarray, torch.Tensor]]]:
        if isinstance(points, np.ndarray):
            points = to_tensor(points)
            return_type = 'numpy'
        else:
            return_type = 'torch'
        origin = to_tensor(origin, device=points.device)
        size = to_tensor(size, device=points.device, dtype=torch.int)
        spacing = to_tensor(spacing, device=points.device)

        # Forward transformed points could end up off-screen and should be filtered.
        # However, we need to know which points are returned for loss calc for example.
        if filter_offscreen:
            assert origin is not None
            assert size is not None
            assert spacing is not None
            # Get crop centre.
            centre_mm = to_tensor([o + (sz * sp) / 2 if c == 'centre' else c for c, sz, sp, o in zip(self.__centre, size, spacing, origin)], device=points.device).T

            # Get crop box.
            crop_min_mm = centre_mm - self.__crop_margin[:, 0].to(points.device)
            crop_max_mm = centre_mm + self.__crop_margin[:, 0].to(points.device)

            # Convert to voxels.
            crop_min_vox = torch.floor((crop_min_mm - origin) / spacing).type(torch.int)
            crop_max_vox = torch.ceil((crop_max_mm - origin) / spacing).type(torch.int) + 1     # Maintain python standard of exclusive upper bound.

            # Truncate to true voxel coords.
            crop_min_vox = torch.clamp(crop_min_vox, 0)
            crop_max_vox = torch.clamp(crop_max_vox, max=size)

            # Convert back to mm.
            crop_min_mm = crop_min_mm * spacing + origin
            crop_max_mm = crop_max_mm * spacing + origin

            # Crop points.
            fov = torch.stack([crop_min_mm, crop_max_mm]).to(points.device)
            to_keep = (points >= fov[0]) & (points < fov[1])
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
