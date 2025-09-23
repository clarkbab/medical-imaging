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

### FovTransforms

- These are easy to implement, we just add/remove voxels.
- How do they behave in a pipeline?
    - When a fov transform (crop/pad) occurs, it affects all downstream
    transformations unless another crop/pad happens.
    - We really need to do a forward pass of all these fov transforms to 
    get the final fov. This fov can then be used to calculate the resampling
    grid points for `back_transform_points`.
    - So what methods do FovTransforms need? Pipeline will benefit from them
    having a transform_fov method that can be call for all transforms in order.

## Single resample

How do we achieve this property?

- Some transforms move points/objects around within an image (`SpatialTransform`).
- These transforms typically map resample points from fixed -> moving image.
- May knock resampling points off-grid and require interolation - increases image artifacts and removes high-frequency information (references?).
- When chaining transforms, more and more information is lost.
- Our `Pipeline` propagates resampling grid points from fixed -> moving image through all `SpatialTransform` objects in reverse order using
`back_transform_points` method before performing a single resampling.
- `FovTransform` objects (e.g. crop/pad) in a `Pipeline` don't move points off-grid, but may increase or decrease the number of resampling grid points back-transformed.

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