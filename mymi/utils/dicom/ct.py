from datetime import datetime
import numpy as np
import os
import pydicom as dcm

from mymi.constants import *
from mymi.typing import *

def from_ct_dicoms(
    cts: List[CtDicom] = [],
    assert_consistency: bool = True,
    dirpath: Optional[str] = None) -> Tuple[CtImageArray, Spacing3D, Point3D]:
    # Load from dirpath if present.
    if dirpath is not None:
        cts = [dcm.dcmread(os.path.join(dirpath, f), force=False) for f in os.listdir(dirpath) if f.endswith('.dcm')]

    # Check CT consistency.
    if assert_consistency:
        # TODO: this doesn't work - xy_pos is a list of tuples.
        xy_pos = [c.ImagePositionPatient[:2] for c in cts]
        xy_pos = round(xy_pos, tol=TOLERANCE_MM)
        xy_pos = np.unique(xy_pos)
        if len(xy_pos) > 1:
            raise ValueError(f"CT slices have inconsistent 'ImagePositionPatient' x/y values: {xy_pos}.")
        z_pos = list(sorted([c.ImagePositionPatient[2] for c in cts]))
        z_pos = round(z_pos, tol=TOLERANCE_MM)
        z_diffs = np.diff(z_pos)
        z_diffs = np.unique(z_diffs)
        if len(z_diffs) > 1:
            raise ValueError(f"CT slices have inconsistent 'ImagePositionPatient' z spacings: {z_diffs}.")

    # Sort CTs by z position, smallest first.
    cts = list(sorted(cts, key=lambda c: c.ImagePositionPatient[2]))

    # Calculate offset.
    # Indexing checked that all 'ImagePositionPatient' keys were the same for the series.
    offset = cts[0].ImagePositionPatient
    offset = tuple(float(o) for o in offset)

    # Calculate size.
    # Indexing checked that CT slices had consisent x/y spacing in series.
    size = (
        cts[0].pixel_array.shape[1],
        cts[0].pixel_array.shape[0],
        len(cts)
    )

    # Calculate spacing.
    # Indexing checked that CT slices were equally spaced in z-dimension.
    spacing = (
        float(cts[0].PixelSpacing[0]),
        float(cts[0].PixelSpacing[1]),
        float(np.abs(cts[1].ImagePositionPatient[2] - cts[0].ImagePositionPatient[2]))
    )

    # Create CT data - sorted by z-position.
    data = np.zeros(shape=size)
    for i, c in enumerate(cts):
        # Convert values to HU.
        slice_data = np.transpose(c.pixel_array)      # 'pixel_array' contains row-first image data.
        slice_data = c.RescaleSlope * slice_data + c.RescaleIntercept

        # Add slice data.
        data[:, :, i] = slice_data

    return data, spacing, offset

def to_ct_dicoms(
    data: CtImageArray, 
    spacing: Spacing3D,
    offset: Point3D,
    pat_id: PatientID,
    study_id: StudyID,
    pat_name: Optional[str] = None,
    series_id: Optional[SeriesID] = None) -> List[CtDicom]:
    pat_name = pat_id if pat_name is None else pat_name
    series_id = f'CT ({study_id})' if series_id is None else series_id

    # Data settings.
    if data.min() < -1024:
        raise ValueError(f"Min CT value {data.min()} is less than -1024. Cannot use unsigned 16-bit values for DICOM.")
    rescale_intercept = -1024
    rescale_slope = 1
    n_bits_alloc = 16
    n_bits_stored = 12
    numpy_type = np.uint16  # Must match 'n_bits_alloc'.
    
    # DICOM data is stored using unsigned int with min=0 and max=(2 ** n_bits_stored) - 1.
    # Don't crop at the bottom, but crop large CT values to be below this threshold.
    ct_max_rescaled = 2 ** (n_bits_stored) - 1
    ct_max = (ct_max_rescaled * rescale_slope) + rescale_intercept
    data = np.minimum(data, ct_max)

    # Perform rescale.
    data_rescaled = (data - rescale_intercept) / rescale_slope
    data_rescaled = data_rescaled.astype(numpy_type)
    scaled_ct_min, scaled_ct_max = data_rescaled.min(), data_rescaled.max()
    if scaled_ct_min < 0 or scaled_ct_max > (2 ** n_bits_stored - 1):
        # This should never happen now that we're thresholding raw HU values.
        raise ValueError(f"Scaled CT data out of bounds: min {scaled_ct_min}, max {scaled_ct_max}. Max allowed: {2 ** n_bits_stored - 1}.")

    # Create study and series fields.
    # StudyID and StudyInstanceUID are different fields.
    # StudyID is a human-readable identifier, while StudyInstanceUID is a unique identifier.
    study_uid = dcm.uid.generate_uid()
    series_uid = dcm.uid.generate_uid()
    frame_of_reference_uid = dcm.uid.generate_uid()
    dt = datetime.now()

    # Create a file for each slice.
    n_slices = data.shape[2]
    ct_dicoms = []
    for i in range(n_slices):
        # Create metadata header.
        file_meta = dcm.dataset.FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 204
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.MediaStorageSOPClassUID = dcm.uid.CtImageArraytorage
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

        # Create DICOM dataset.
        ct_dicom = dcm.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
        ct_dicom.is_little_endian = True
        ct_dicom.is_implicit_VR = True
        ct_dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ct_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

        # Set other required fields.
        ct_dicom.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
        ct_dicom.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)
        ct_dicom.InstitutionName = 'PMCC'
        ct_dicom.Manufacturer = 'PMCC'
        ct_dicom.Modality = 'CT'
        ct_dicom.SpecificCharacterSet = 'ISO_IR 100'

        # Add patient info.
        ct_dicom.PatientID = pat_id
        ct_dicom.PatientName = pat_name

        # Add study info.
        ct_dicom.StudyDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.StudyDescription = study_id
        ct_dicom.StudyInstanceUID = study_uid
        ct_dicom.StudyID = study_id
        ct_dicom.StudyTime = dt.strftime(DICOM_TIME_FORMAT)

        # Add series info.
        ct_dicom.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
        ct_dicom.SeriesDescription = series_id,
        ct_dicom.SeriesInstanceUID = series_uid
        ct_dicom.SeriesNumber = 0
        ct_dicom.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)

        # Add data.
        ct_dicom.BitsAllocated = n_bits_alloc
        ct_dicom.BitsStored = n_bits_stored
        ct_dicom.FrameOfReferenceUID = frame_of_reference_uid
        ct_dicom.HighBit = 11
        offset_z = offset[2] + i * spacing[2]
        ct_dicom.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ct_dicom.ImagePositionPatient = [offset[0], offset[1], offset_z]
        ct_dicom.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ct_dicom.InstanceNumber = i + 1
        ct_dicom.PhotometricInterpretation = 'MONOCHROME2'
        ct_dicom.PatientPosition = 'HFS'
        ct_dicom.PixelData = np.transpose(data_rescaled[:, :, i]).tobytes()   # Uses (y, x) spacing.
        ct_dicom.PixelRepresentation = 0
        ct_dicom.PixelSpacing = [spacing[0], spacing[1]]    # Uses (x, y) spacing.
        ct_dicom.RescaleIntercept = rescale_intercept
        ct_dicom.RescaleSlope = rescale_slope
        ct_dicom.Rows, ct_dicom.Columns = data.shape[1], data.shape[0]
        ct_dicom.SamplesPerPixel = 1
        ct_dicom.SliceThickness = float(abs(spacing[2]))

        ct_dicoms.append(ct_dicom)

    return ct_dicoms
