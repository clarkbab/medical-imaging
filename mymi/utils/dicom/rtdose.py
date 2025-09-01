from datetime import datetime
import pydicom as dcm
import numpy as np

from mymi.constants import *
from mymi.typing import *

def from_rtdose_dicom(rtdose_dicom: RtDoseDicom) -> Tuple[DoseImageArray, Spacing3D, Point3D]:
    # Load data.
    data = np.transpose(rtdose_dicom.pixel_array)
    data = rtdose_dicom.DoseGridScaling * data

    # Load spacing.
    spacing_xy = rtdose_dicom.PixelSpacing 
    z_diffs = np.diff(rtdose_dicom.GridFrameOffsetVector)
    z_diffs = round(z_diffs, tol=TOLERANCE_MM)
    z_diffs = np.unique(z_diffs)
    if len(z_diffs) != 1:
        raise ValueError(f"Slice z spacings for RtDoseDicom not equal: {z_diffs}.")
    spacing_z = z_diffs[0]
    spacing = tuple((float(s) for s in np.append(spacing_xy, spacing_z)))

    # Get offset.
    offset = tuple(float(o) for o in rtdose_dicom.ImagePositionPatient)

    return data, spacing, offset

def to_rtdose_dicom(
    data: DoseImageArray, 
    spacing: Spacing3D,
    offset: Point3D,
    grid_scaling: float = 1e-3,
    ref_ct: Optional[CtDicom] = None,
    rtdose_template: Optional[RtDoseDicom] = None,
    series_desc: Optional[str] = None) -> RtDoseDicom:
    if rtdose_template is not None:
        # Start from the template.
        rtdose_dicom = rtdose_template.copy()

        # Overwrite sop ID.
        file_meta = rtdose_dicom.file_meta.copy()
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        rtdose_dicom.file_meta = file_meta
        rtdose_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    else:
        # Create rtdose from scratch.
        file_meta = dcm.dataset.Dataset()
        file_meta.FileMetaInformationGroupLength = 204
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
        file_meta.MediaStorageSOPClassUID = dcm.uid.RTDoseStorage
        file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
        file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

        rtdose_dicom = dcm.dataset.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
        rtdose_dicom.BitsAllocated = 32
        rtdose_dicom.BitsStored = 32
        rtdose_dicom.DoseGridScaling = grid_scaling
        rtdose_dicom.DoseSummationType = 'PLAN'
        rtdose_dicom.DoseType = 'PHYSICAL'
        rtdose_dicom.DoseUnits = 'GY'
        rtdose_dicom.HighBit = 31
        rtdose_dicom.Modality = 'RTDOSE'
        rtdose_dicom.PhotometricInterpretation = 'MONOCHROME2'
        rtdose_dicom.PixelRepresentation = 0
        rtdose_dicom.SamplesPerPixel = 1
        rtdose_dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
        rtdose_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    # Set custom attributes.
    rtdose_dicom.DeviceSerialNumber = ''
    rtdose_dicom.InstitutionAddress = ''
    rtdose_dicom.InstitutionName = 'PMCC'
    rtdose_dicom.InstitutionalDepartmentName = 'PMCC-AI'
    rtdose_dicom.Manufacturer = 'PMCC-AI'
    rtdose_dicom.ManufacturerModelName = 'PMCC-AI'
    rtdose_dicom.SoftwareVersions = ''
    
    # Copy atributes from reference ct/rtdose dicom.
    assert rtdose_template is not None or ref_ct is not None
    ref_dicom = rtdose_template if rtdose_template is not None else ref_ct
    attrs = [
        'AccessionNumber',
        'FrameOfReferenceUID',
        'PatientBirthDate',
        'PatientID',
        'PatientName',
        'PatientSex',
        'StudyDate',
        'StudyDescription',
        'StudyID',
        'StudyInstanceUID',
        'StudyTime'
    ]
    for a in attrs:
        if hasattr(ref_dicom, a):
            setattr(rtdose_dicom, a, getattr(ref_dicom, a))

    # Add series info.
    series_desc = rtdose_dicom.StudyID if series_desc is None else series_desc
    rtdose_dicom.SeriesDescription = f'RTDOSE ({series_desc})'
    rtdose_dicom.SeriesInstanceUID = dcm.uid.generate_uid()
    rtdose_dicom.SeriesNumber = 1

    # Remove some attributes that might be set from the template.
    remove_attrs = [
        'OperatorsName',
        'StationName',
    ]
    if rtdose_template is not None:
        for a in remove_attrs:
            if hasattr(rtdose_dicom, a):
                delattr(rtdose_dicom, a)

    # Set image properties.
    rtdose_dicom.Columns = data.shape[0]
    rtdose_dicom.FrameIncrementPointer = dcm.datadict.tag_for_keyword('GridFrameOffsetVector')
    rtdose_dicom.GridFrameOffsetVector = [i * spacing[2] for i in range(data.shape[2])]
    rtdose_dicom.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    rtdose_dicom.ImagePositionPatient = list(offset)
    rtdose_dicom.ImageType = ['DERIVED', 'SECONDARY', 'AXIAL']
    rtdose_dicom.NumberOfFrames = data.shape[2]
    rtdose_dicom.PixelSpacing = [spacing[0], spacing[1]]    # Uses (x, y) spacing.
    rtdose_dicom.Rows = data.shape[1]
    rtdose_dicom.SliceThickness = spacing[2]

    # Get grid scaling and data type.
    grid_scaling = rtdose_dicom.DoseGridScaling
    n_bits = rtdose_dicom.BitsAllocated
    if n_bits == 16:
        data_type = np.uint16
    elif n_bits == 32:
        data_type = np.uint32
    else:
        raise ValueError(f'Unsupported BitsAllocated value: {n_bits}. Must be 16 or 32.')

    # Add dose data. 
    data = (data / grid_scaling).astype(data_type)
    rtdose_dicom.PixelData = np.transpose(data).tobytes()     # Uses (z, y, x) format.

    # Set timestamps.
    dt = datetime.now()
    rtdose_dicom.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose_dicom.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose_dicom.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)

    return rtdose_dicom
