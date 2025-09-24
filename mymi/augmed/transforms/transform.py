from typing import *

from mymi.typing import *
from mymi.utils import alias_kwargs, arg_to_list

# What is a Transform?
# Transform defines the API that any (deterministic) Transform
# and RandomTransform must follow.
# What about pipeline? Yeah, I guess so. We treat it just like a transform.
class Transform:
    @alias_kwargs(('uic', 'use_image_coords'))
    def __init__(
        self,
        dim: SpatialDim = 3,
        use_image_coords: bool = False) -> None:
        assert dim in [2, 3], "Only 2D and 3D flips are supported."
        self._dim = dim
        self._use_image_coords = use_image_coords
    
    # Alias for 'transform' method.
    def __call__(
        self,
        data: Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]],
        # Require this ordering of kwargs for API simplicity.
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        **kwargs) -> Union[ImageArray, ImageTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, PointsArray, PointsTensor]]]:
        return self.transform(data, origin=origin, spacing=spacing, **kwargs)

    @property
    def dim(self) -> SpatialDim:
        return self._dim

    @property
    def params(self) -> Dict[str, Any]:
        if not hasattr(self, '_params'):
            raise ValueError("Subclasses of 'Transform' must have '_params' attribute.")
        return self._params

    def __repr__(self) -> str:
        return str(self)

    def set_dim(
        self,
        dim: SpatialDim) -> None:
        assert dim in [2, 3], "Only 2D and 3D transforms are supported."
        self._dim = dim

    def __str__(
        self,
        class_name: str,
        params: Dict[str, str]) -> str:
        params['dim'] = self._dim
        params['use_image_coords'] = self._use_image_coords
        return f"{class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"

    # Originally this was defined as a mixin to avoid having RandomTransforms override the method.
    # However, as a mixin, each new transform class needs to subclass the mixin also, which creates
    # more boilerplate for new transforms.
    @alias_kwargs([
        ('o', 'origin'),
        ('s', 'spacing'),
    ])
    # Can pass a single array/tensor or a list of arrays/tensors.
    # Points arrays/tensors are inferred by their Nx2/3 shape. It's unlikely that images of this size will
    # be passed, but it would break.
    # Labels are inferred by the data type of the passed array/tensor (bool) and will be returned
    # in boolean type.
    # Will return a single transformed array/tensor or list of arrays/tensors.
    # If a single spacing/origin/size is passed, this is broadcast to all image arrays/tensors,
    # other
    def transform(
        self,
        data: Union[ImageArray, ImageTensor, LabelArray, LabelTensor, PointsArray, PointsTensor, List[Union[ImageArray, ImageTensor, LabelArray, LabelTensor, PointsArray, PointsTensor]]],
        spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
        origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
        # This comes last because it can be inferred via adjacent images.
        size: Optional[Union[Size, SizeArray, SizeTensor, List[Union[Size, SizeArray, SizeTensor]]]] = None,
        filter_offgrid: bool = True,
        return_grid: bool = False) -> Union[ImageArrayWithFov, ImageTensorWithFov, PointsArray, PointsTensor, List[Union[ImageArrayWithFov, ImageTensorWithFov, PointsArray, PointsTensor]]]:
        datas, data_was_single = arg_to_list(data, (np.ndarray, torch.Tensor), return_matched=True)
        sizes = arg_to_list(size, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(datas))
        spacings = arg_to_list(spacing, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(datas))
        origins = arg_to_list(origin, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(datas))

        # Infer data types.
        image_indices = []
        points_indices = []
        data_types = {}
        for i, d in enumerate(datas):
            if d.shape[-1] == 2 or d.shape[-1] == 3:
                points_indices.append(i)
                data_types[i] = 'points'
            else:
                image_indices.append(i)
                data_types[i] = 'image'

        # Infer sizes for offscreen point filtering.
        if filter_offgrid:
            for i in points_indices:
                if sizes[i] is None:
                    # Infer size from images - must all have same shape.
                    image_sizes = [datas[j].shape[-self._dim:] for j in image_indices]
                    if len(image_sizes) > 0 and np.unique(image_sizes, axis=0).shape[0] == 1:
                        sizes[i] = tuple(image_sizes[0])

        # Transform images.
        images, image_spacings, image_origins = [datas[i] for i in image_indices], [spacings[i] for i in image_indices], [origins[i] for i in image_indices]
        print('transform')
        print(return_grid)
        images_ts = self.transform_image(images, spacing=image_spacings, origin=image_origins, return_grid=return_grid)

        # Transform points.
        points, points_sizes, points_spacings, points_origins = [datas[i] for i in points_indices], [sizes[i] for i in points_indices], [spacings[i] for i in points_indices], [origins[i] for i in points_indices]
        points_ts = []
        for p, sz, sp, o in zip(points, points_sizes, points_spacings, points_origins):
            if sz is None:
                filter_offgrid = False    # Only filter if 'size' was passed or inferred from image sizes.
            points_t = self.transform_points(p, filter_offgrid=filter_offgrid, size=sz, spacing=sp, origin=o)
            points_ts.append(points_t)

        # Flatten results.
        datas_t = []
        image_i, points_i = 0, 0
        for i in range(len(datas)):
            if data_types[i] == 'image':
                datas_t.append(images_ts[image_i])
                image_i += 1
            else:
                datas_t.append(points_ts[points_i])
                points_i += 1

        return datas_t[0] if data_was_single else datas_t

    def transform_image(
        self,
        image: Union[ImageArray, ImageTensor],
        **kwargs) -> Union[ImageArray, ImageTensor]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_image' method.")

    def transform_points(
        self,
        points: Union[PointsArray, PointsTensor],
        **kwargs) -> Union[PointsArray, PointsTensor]:
        raise ValueError("Subclasses of 'Transform' must implement 'transform_points' method.")
