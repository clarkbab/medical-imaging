import numpy as np
import SimpleITK as sitk
import struct

from mymi.types import PointMM3D
from mymi.utils import to_sitk, transpose_image

def velocity_load_transform(
    filepath: str,
    # Velocity '.bdf' file does not preserve the DICOM 'ImageOffsetPatient' offset.
    # We need to pass this manually.
    fixed_offset: PointMM3D) -> sitk.Transform:

    # Read ".bdf" file.
    with open(filepath, "rb") as f:
        data = f.read()
        
    # Read transform image size.
    # Read data as "sliding windows" of bytes.
    # Data format is 32-bit unsigned int (I), little-endian (<).
    size = []
    n_dims = 3
    n_bytes_per_val = 4
    n_bytes = n_dims * n_bytes_per_val
    data_format = "<I"
    for i in range(0, n_bytes, n_bytes_per_val):
        size_i = struct.unpack(data_format, data[i:i + n_bytes_per_val])[0]
        size.append(size_i)
    size = tuple(size)

    # Read transform image pixel spacing.
    # Data format is 32-bit float (f).
    spacing = []
    data_format = "f"
    start_byte = 12
    n_dims = 3
    n_bytes = n_dims * n_bytes_per_val
    for i in range(start_byte, start_byte + n_bytes, n_bytes_per_val):
        spacing_i = struct.unpack(data_format, data[i:i + n_bytes_per_val])[0]
        spacing.append(spacing_i)
    spacing = tuple(spacing)

    # Sanity check number of bytes in file.
    # Should be num. voxels * 3 * 4 (each voxel is a 3 dimensional, 32-bit float) + 24 bytes for image size and spacing header.
    n_voxels = np.prod(size)
    n_bytes = len(data)
    n_bytes_expected = n_voxels * n_bytes_per_val * n_dims + 24
    if n_bytes != n_bytes_expected:
        raise ValueError(f"File '{filepath}' should contain '{n_bytes_expected}' bytes (num. voxels ({n_voxels}) * 4 bytes * 3 axes + 24 bytes header), got '{n_bytes}'.")

    # Read vector image.
    vector = []
    image = []
    start_byte = 24
    n_bytes = n_voxels * n_dims * n_bytes_per_val
    for i in range(start_byte, start_byte + n_bytes, n_bytes_per_val):
        vector_i = struct.unpack(data_format, data[i:i + n_bytes_per_val])[0]
        vector.append(vector_i)
        if (i - (start_byte - n_bytes_per_val)) % n_dims * n_bytes_per_val == 0:
            image.append(vector.copy())
            vector = []
            
    if len(image) != n_voxels:
        raise ValueError(f"Expected image to contain '{n_voxels}' voxels, got '{len(image)}'.")

    # We loaded a flat array. This flat array represents a 3D image stored in column-major order (as used
    # by Velocity) - meaning the first elements belong to the first column (0th axis). Numpy uses row-major 
    # ordering, which means that when performing 'np.reshape', the first elements will be put into the first
    # row (or 2nd axis for 3D image). We can get the intended image by reshaping using reversed axes (z, y, x)
    # and then taking the transpose to get (x, y, z).
    image = np.array(image)
    image = np.reshape(image, (*reversed(size), 3))
    image = transpose_image(image, is_vector=True)

    # Create transform.
    # The 'offset' is not stored in the '.bdf' file, so we need to use the fixed image offset.
    image = to_sitk(image, spacing, fixed_offset, is_vector=True)
    transform = sitk.DisplacementFieldTransform(image)
    return transform
