# Augmed

## Installation

```
pip install augmed
```

## Motivation

What properties are desirable for a medical imaging data augmentation library?

- **GPU-accelerated**: GPU support provided for all transform classes.\
Using GPUs for data augmentation decreases training times and GPU idle times (references?), particularly\
when using complex transforms like elastic deformation (references?).
- **Point transforms**. Point transforms provided for all transform classes.\
Image registration models benefit from the inclusion of anatomical landmarks during training (references?).\
However, existing augmentation libraries only provide image augmentation methods. Augmed provides point (forward) \
transforms, in addition to image (backward) transforms for all transform classes. This includes elastic deformation,\
where invertibility can be ensured by constraining the magnitude of random displacements.
- **Single resample**. Augmed performs a single resample for all pipline transforms.\
Performing sequential image resampling increases image artifacts and reduces high-frequency components (references?).\
When performing sequential data augmentations (e.g. elastic -> affine -> crop) it would be beneficial to perform a single resampling operation.
- **2/3D image support**: Self-explanatory. References?
- **Implicit API**. Rather than defining data types, use existing types for images and points and let the transform figure\
out the types based on the input shapes and data types.

## Transforms

### GridTransforms

- These are easy to implement, we just add/remove voxels.
- How do they behave in a pipeline?
    - When a grid transform (crop/pad) occurs, it affects all downstream
    transformations unless another crop/pad happens.
    - We really need to do a forward pass of all these grid transforms to 
    get the final grid. This grid can then be used to calculate the resampling
    grid points for `back_transform_points`.
    - So what methods do GridTransforms need? Pipeline will benefit from them
    having a transform_grid method that can be call for all transforms in order.

## Single resample

How do we achieve this property?

- Some transforms move points/objects around within an image (`SpatialTransform`).
- These transforms typically map resample points from fixed -> moving image.
- May knock resampling points off-grid and require interolation - increases image artifacts and removes high-frequency information (references?).
- When chaining transforms, more and more information is lost.
- Our `Pipeline` propagates resampling grid points from fixed -> moving image through all `SpatialTransform` objects in reverse order using
`back_transform_points` method before performing a single resampling.
- `GridTransform` objects (e.g. crop/pad) in a `Pipeline` don't move points off-grid, but may increase or decrease the number of resampling grid points back-transformed.

## Point transforms

How do we achieve this property?

- For `SpatialTransform` objects, we define a 'transform_points' method that is the inverse of `back_transform_points`.
- For elastic deformation, an inverse is ensured by keeping displacement magnitudes to less than half the control grid
spacing. Inverse is calculated using a GPU-accelerated iterative method.

## Performance tweaks

- Chained affine transforms are resolved by successive 4x4 matrix multiplications (homogeneous coords) before applying to large tensor of grid points (Nx4).

## Library comparison

Which MONAI transforms support a torch backend? 

| Library     | 3D image support | GPU support | Single resample | Point transforms |
| :-------    | :------: | :------: | -------: | ------: |
| augmed      | &#x2713; | &#x2713; | &#x2713; | &#x2713; |
| monai       | &#x2713; | Partial | Partial ([affine only](https://docs.monai.io/en/latest/transforms.html#lazytrait)) |
| torchio     | &#x2713; | [&#x2717;](https://github.com/TorchIO-project/torchio/issues/388)  | [&#x2717;](https://github.com/TorchIO-project/torchio/blob/8065c45838ce92a0bbddb5f6b65319ea93b7deaa/src/torchio/transforms/augmentation/composition.py#L55) | [&#x2717;](https://github.com/TorchIO-project/torchio/issues/1274)
| torchvision | &#x2717; | &#x2713; |

## Transform types

- Spatial transforms: Move the positions of objects in the image.
- Intensity transforms: Change the appearance of objects in the image.
- Field-of-view transforms: Change the view window size and position (e.g. crop/pad). Sampling (draw random patches from the image), Patching (draw a batch of regularly spaced samples from the image).

## Transform API

- 3-5D arrays or tensors (N, C, X, Y, Z) are accepted with 2-3 spatial dimensions and optional batch/channel. The spatial
dimension is determined through the 'dim' parameter shared by all transforms.
- Types (image vs. points, intensity vs. label image) are determined by the passed array/tensor shapes and types.
- When are size/spacing/origin needed during transforms?
    - TODO: Show examples of where all this goes wrong!
    - Spacing/origin default to 1mm and 0mm respectively - which is equivalent to voxel coords.
    - Image transforms do not require spacing but transforms will not be accurate if images have isotropic spacing and spacing is not passed. TODO: Show incorrect rotation because non-iso spacing wasn't passed.
    - Image transforms do not require origin except when transforming images and points to ensure points and image objects are transformed correctly. TODO: Show separate transforming of points and images where origin wasn't set on the image, and images/points were transformed differently.
    - Size is inferred for image transforms from the passed image size.
    - Size is not required for points transforms if we don't want to filter off-grid points after transformation, or if points are transformed alongside image/s - as size can be inferred as long as all images have the same size. TODO: Show error when points without image, or points with images of different sizes.

## Range API

For many random transforms we have to define ranges of values.

For example, how much to crop from an image?
We define this as a tuple that tells us how much to remove from each end of each axis.
E.g. crop=(80, 120, 80, 120, ...) tells us to remove between 80-120mm from each end of
the x-axis. These ranges can be asymmetric of course, if we'd like to remove more from one
end of the image -> e.g. crop=(0, 0, 80, 120) won't remove anything from the lower end of
the x-axis.
So what if we would like symmetric crops?
We might define crop=(80, 120, 80, 120, ...), but we'd actually like x (80 <= x <= 120) to
match on both ends. We should have a 'symmetric: bool' parameter that controls this.
Mostly, people will be passing crop=(80, 120) which will expand to symmetric ranges, no problem.
If someone passes an asymmetric range, e.g. crop=(0, 0, 80, 120) but requests 'symmetric=True'
we should raise an error.

## Monitoring errors
- Opt-in telemetry for raised exceptions?
- "report_error" block that will catch and send exception info?

## Patient or voxel coords.
- All values are specified in patient coordinates (mm).
- Should we provide an option for voxels? E.g. what if someone wants to crop 10-20 voxels from
an image? They would need to calculate this value in mm before applying to the image, and then rounding, which for cropping and padding will favour adding voxels, might mean this value is wrong. This shouldn't be the case.
- If someone uses spacing=1mm, then crop=10mm will be the same as 10 voxels. But someone might want to
crop 10 voxels, for a spacing=2mm or other image.
- Each transform should have a "use_image_coords: bool = False" parameter that can be set to false if passed
parameters are in image coordinates. Parameters will be converted to mm internally.