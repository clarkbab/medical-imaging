from datetime import datetime
import pydicom as dcm

from mymi.constants import *
from mymi.typing import *

def from_ct_dicoms(cts: List[CtDicom]) -> Tuple[CtData, Spacing3D, Point3D]:
    pass

def to_ct_dicoms(
    data: CtData, 
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
        file_meta.MediaStorageSOPClassUID = dcm.uid.CTImageStorage
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
