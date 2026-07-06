import SimpleITK as sitk

def compute_affine_transform_2d(fixed_points, moving_points):
    """
    Compute the affine transform mapping fixed points to moving points
    using SimpleITK's landmark-based initializer.

    Parameters
    ----------
    fixed_points : list of (x, y) tuples — 3 points in the fixed image
    moving_points : list of (x, y) tuples — 3 points in the moving image

    Returns
    -------
    sitk.AffineTransform (2D) such that transform(fixed_point) ≈ moving_point
    """
    assert len(fixed_points) == len(moving_points) == 3, \
        "Need exactly 3 point correspondences for 2D affine."

    # SimpleITK expects flat lists: [x1, y1, x2, y2, x3, y3]
    fixed_flat = [coord for point in fixed_points for coord in point]
    moving_flat = [coord for point in moving_points for coord in point]

    # Initialize a 2D affine transform
    affine = sitk.AffineTransform(2)

    transform = sitk.LandmarkBasedTransformInitializer(
        affine,
        fixed_flat,
        moving_flat
    )

    # Cast back to AffineTransform (initializer returns the same type given)
    affine_transform = sitk.AffineTransform(transform)

    return affine_transform


if __name__ == "__main__":
    fixed_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    moving_points = [(5.0, 5.0), (5.0, 7.0), (3.0, 5.0)]

    tx = compute_affine_transform_2d(fixed_points, moving_points)

    print("Matrix:", tx.GetMatrix())
    print("Translation:", tx.GetTranslation())
    print("Center:", tx.GetCenter())

    # Verify: transform should map each fixed point to its moving counterpart
    for fp, mp in zip(fixed_points, moving_points):
        mapped = tx.TransformPoint(fp)
        print(f"{fp} -> {mapped} (expected {mp})")
