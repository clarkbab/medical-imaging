import numpy as np
import SimpleITK as sitk
import struct

from mymi import logging
from mymi.typing import *
from mymi.utils import *

def load_velocity_bdf_transform(
    filepath: str,
    # Velocity '.bdf' file does not preserve the DICOM 'ImageOffsetPatient' origin.
    # We need to pass this manually.
    fixed_origin: Point3D) -> sitk.Transform:
    logging.info(f"Loading velocity transform at: {filepath}")

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
    image = np.moveaxis(image, -1, 0)
    image = transpose_image(image, vector=True)

    # Create transform.
    # The 'origin' is not stored in the '.bdf' file, so we need to use the fixed image origin.
    image = to_sitk_image(image, spacing=spacing, origin=fixed_origin, vector=True)
    transform = sitk.DisplacementFieldTransform(image)

    # Is rigid transform baked into the BDF file??

    return transform

def load_velocity_transform(filepath: FilePath) -> sitk.Transform:
    logging.info(f"Loading velocity REG transform at: {filepath}")

    # Load REG dicom.
    dicom = dcm.dcmread(filepath)
    assert dicom.Modality == 'REG'

    if hasattr(dicom, 'RegistrationSequence'):  # Rigid.
        logging.info("Loading velocity rigid transform.")
        assert not hasattr(dicom, 'DeformableRegistrationSequence')
        reg_seq = dicom.RegistrationSequence
        assert len(reg_seq) == 2

        # Check second transform - should be identity.
        reg_2 = reg_seq[1]
        reg_2_mat_rs = reg_2.MatrixRegistrationSequence
        assert len(reg_2_mat_rs) == 1
        reg_2_mat_r = reg_2_mat_rs[0]
        reg_2_mats = reg_2_mat_r.MatrixSequence
        assert len(reg_2_mats) == 1
        reg_2_mat = reg_2_mats[0]
        assert reg_2_mat.FrameOfReferenceTransformationMatrixType == 'RIGID', reg_2_mat.FrameOfReferenceTransformationMatrixType
        assert reg_2_mat.FrameOfReferenceTransformationMatrix == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], reg_2_mat.FrameOfReferenceTransformationMatrix

        # Get transform.
        rigid_reg = reg_seq[0]
        rigid_reg_mat_rs = rigid_reg.MatrixRegistrationSequence
        assert len(rigid_reg_mat_rs) == 1
        rigid_reg_mat_r = rigid_reg_mat_rs[0]
        rigid_reg_mats = rigid_reg_mat_r.MatrixSequence
        assert len(rigid_reg_mats) == 1
        rigid_reg_mat = rigid_reg_mats[0]
        assert rigid_reg_mat.FrameOfReferenceTransformationMatrixType == 'RIGID', rigid_reg_mat.FrameOfReferenceTransformationMatrixType
        # This affine matrix lines up with that shown in Velocity UI. This is stated to be
        # the secondary -> primary registration matrix.
        affine_matrix = np.array(rigid_reg_mat.FrameOfReferenceTransformationMatrix).reshape((4, 4))
        logging.info(f"Loaded velocity rigid transform with matrix: {affine_matrix}")
        affine_rot = affine_matrix[0:3, 0:3].astype(np.float64)
        affine_translation = affine_matrix[0:3, 3]
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(affine_rot.flatten().tolist())
        transform.SetTranslation(affine_translation)

        # Our workflow uses transforms that operate from fixed -> moving image space.
        transform = transform.GetInverse()

        return transform

    elif hasattr(dicom, 'DeformableRegistrationSequence'):  # Deformable.
        logging.info("Loading velocity deformable transform.")
        assert not hasattr(dicom, 'RegistrationSequence')

        # Load reg sequence.
        reg_seq = dicom.DeformableRegistrationSequence
        assert len(reg_seq) == 2, "Version change?"

        # What is first reg?
        # This reg has a "SourceFrameOfReferenceUID" that matches the "FrameOfReferenceUID" of the 
        # fixed image. Also the "ReferencedImageSequence" refers to the fixed image.
        reg_1 = reg_seq[0]

        # Get affine (pre-deformation) transform.
        # This reg has a "SourceFrameOfReferenceUID" that matches the "FrameOfReferenceUID" of the
        # moving image. Also the "ReferencedImageSequence" refers to the moving image.
        deform_reg = reg_seq[1]
        assert hasattr(deform_reg, 'PreDeformationMatrixRegistrationSequence'), "Version change?"
        pdmr_seq = deform_reg.PreDeformationMatrixRegistrationSequence
        assert len(pdmr_seq) == 1
        pdmr = pdmr_seq[0]
        # This affine matrix DOES NOT line up with that shown in Velocity UI. The affine matrix 
        # stored here is the primary -> secondary transform - this can be validated by checking
        # the inverse (logged below) against Velocity UI. We also verify by visually checking
        # alignment of fixed/moved images after applying the affine transformation for patients
        # with large translations - it'd be obvious if we were performing the wrong (inverse)
        # transform.
        affine_matrix = np.array(pdmr.FrameOfReferenceTransformationMatrix).reshape((4, 4))
        logging.info(f"Loaded velocity rigid transform with matrix: {affine_matrix}")
        affine_rot = affine_matrix[0:3, 0:3].astype(np.float64)
        affine_translation = affine_matrix[0:3, 3]
        affine_transform = sitk.AffineTransform(3)
        affine_transform.SetMatrix(affine_rot.flatten().tolist())
        affine_transform.SetTranslation(affine_translation)
        inv_transform = affine_transform.GetInverse()
        logging.info(f"Validate rigid transform (secondary -> primary) within velocity. Rotation: {inv_transform.GetMatrix()}, Translation: {inv_transform.GetTranslation()}")

        # Get deformable transform.
        # This grid is defined on the fixed image, we can see this by looking at the Velocity UI grid overlay
        # and seeing that the origin is defined in fixed image coordinates. See PMCC_ReIrrad_L14 for example
        # with very different coordinate spaces between moving/fixed images. For this reason, when moving points
        # from fixed -> moving image space, we need to deform before applying affine - otherwise the points will
        # be hitting the deformation off-grid (identity).

        # This is a grid of displacement vectors that has different spacing to the fixed image grid.
        # We know (from Velocity docs) that the registration was created using a b-spline model. So,
        # this is a sampled grid from a continuous b-spline. SimpleITK uses linear interpolation for
        # DisplacementFieldTransforms (vector grids), whereas Velocity might be using the original
        # b-spline model for its point review function.
        # TODO: Try fitting the displacement grid to a b-spline model and see what happens.
        assert hasattr(deform_reg, 'DeformableRegistrationGridSequence'), "Version change?"
        reg_grids = deform_reg.DeformableRegistrationGridSequence
        assert len(reg_grids) == 1
        reg_grid = reg_grids[0]
        grid_size = [int(x) for x in reg_grid.GridDimensions]
        grid_spacing = [float(x) for x in reg_grid.GridResolution]
        grid_origin = [float(x) for x in reg_grid.ImagePositionPatient]
        logging.info(f"Loading displacement grid with size={grid_size}, spacing={grid_spacing}, and origin={grid_origin}.")

        # Read vector image.
        n_dims = len(grid_size)
        n_voxels = np.prod(grid_size)
        n_bytes_per_voxel = 4
        n_bytes = n_dims * n_voxels * n_bytes_per_voxel
        dvf_image = np.frombuffer(reg_grid.VectorGridData, dtype='<f4')
        assert len(dvf_image) == n_dims * n_voxels

        # We loaded a flat array. This flat array represents a 3D image stored in column-major order (as used
        # by Velocity) - meaning the first elements belong to the first column (0th axis). Numpy uses row-major 
        # ordering, which means that when performing 'np.reshape', the first elements will be put into the first
        # row (or 3rd axis for 3D image). We can get the intended image by reshaping using reversed axes (z, y, x)
        # and then taking the transpose to get (x, y, z).
        dvf_image = np.array(dvf_image)
        dvf_image = np.reshape(dvf_image, (*reversed(grid_size), n_dims))
        dvf_image = np.moveaxis(dvf_image, -1, 0)
        dvf_image = transpose_image(dvf_image, vector=True)

        # Create transform.
        dvf_image = to_sitk_image(dvf_image.astype(np.float64), spacing=grid_spacing, origin=grid_origin, vector=True)
        dvf_transform = sitk.DisplacementFieldTransform(dvf_image)

        # Create composite transform.
        # Dicom standard says: First, transform the coordinates using the matrix described in the Pre Deformation Matrix
        # Registration Sequence (0064,000F). I.e. our affine transform.
        transform = sitk.CompositeTransform(3)
        transform.AddTransform(affine_transform)
        transform.AddTransform(dvf_transform)

    else:
        raise ValueError(f"Unrecognised velocity transform type at filepath: {filepath}")

    return transform
