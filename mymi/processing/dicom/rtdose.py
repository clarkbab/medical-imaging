from datetime import datetime
import pydicom as dcm
import numpy as np

from mymi.constants import *
from mymi.typing import *

def to_rtdose_dicom(
    data: DoseData, 
    spacing: Spacing3D,
    offset: Point3D,
    ref_ct: CtDicom,
    series_id: Optional[SeriesID] = None) -> RtDoseDicom:

    # Convert from Gy to cGy for integer storage.
    grid_scaling = 0.01
    data = (data / grid_scaling).astype(np.uint16)  # Stored in cGy
    
    # Create metadata.
    file_meta = dcm.dataset.Dataset()
    file_meta.FileMetaInformationGroupLength = 204
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.MediaStorageSOPClassUID = dcm.uid.RTDoseStorage
    file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
    file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian
    
    # Create RTDOSE dicom.
    rtdose_dicom = dcm.dataset.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
    rtdose_dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
    rtdose_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    rtdose_dicom.Modality = 'RTDOSE'
    rtdose_dicom.Manufacturer = 'MYMI'
    
    # Copy patient info.
    attrs = ['AccessionNumber', 'PatientBirthDate', 'PatientID', 'PatientName', 'PatientSex', 'SeriesInstanceUID', 'StudyDate','StudyInstanceUID', 'StudyTime']
    for a in attrs:
        if hasattr(ref_ct, a):
            setattr(rtdose_dicom, a, getattr(ref_ct, a))

    # Copy frame of reference.
    rtdose_dicom.FrameOfReferenceUID = ref_ct.FrameOfReferenceUID

    # Set image properties.
    rtdose_dicom.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    rtdose_dicom.ImagePositionPatient = list(offset)
    rtdose_dicom.PixelSpacing = [spacing[0], spacing[1]]    # Uses (x, y) spacing.
    rtdose_dicom.SliceThickness = spacing[2]
    rtdose_dicom.GridFrameOffsetVector = [i * spacing[2] for i in range(data.shape[2])]
    rtdose_dicom.NumberOfFrames = data.shape[2]
    rtdose_dicom.Rows = data.shape[1]
    rtdose_dicom.Columns = data.shape[0]
    
    # Add frame of reference.
    ref_ct_series = dcm.dataset.Dataset()
    ref_ct_series.SeriesInstanceUID = ref_ct.SeriesInstanceUID
    ref_study = dcm.dataset.Dataset()
    ref_study.ReferencedSOPClassUID = ref_ct.SOPClassUID
    ref_study.ReferencedSOPInstanceUID = ref_ct.StudyInstanceUID
    ref_study.RTReferencedSeriesSequence = [ref_ct_series]
    ref_frame = dcm.dataset.Dataset()
    ref_frame.FrameOfReferenceUID = ref_ct.FrameOfReferenceUID
    ref_frame.RTReferencedStudySequence = [ref_study]
    rtdose_dicom.ReferencedFrameOfReferenceSequence = [ref_frame]

    # Add study info - from ref CT.
    rtdose_dicom.StudyDate = ref_ct.StudyDate
    rtdose_dicom.StudyDescription = getattr(ref_ct, 'StudyDescription', '')
    rtdose_dicom.StudyID = ref_ct.StudyID
    rtdose_dicom.StudyInstanceUID = ref_ct.StudyInstanceUID
    rtdose_dicom.StudyTime = ref_ct.StudyTime

    # Add series info.
    series_id = f'RTDOSE ({ref_ct.StudyID})' if series_id is None else series_id
    rtdose_dicom.SeriesDescription = series_id,
    rtdose_dicom.SeriesInstanceUID = dcm.uid.generate_uid()
    rtdose_dicom.SeriesNumber = 1

    # Dose settings.
    rtdose_dicom.DoseUnits = 'GY'
    rtdose_dicom.DoseType = 'PHYSICAL'
    rtdose_dicom.DoseSummationType = 'PLAN'
    rtdose_dicom.DoseGridScaling = grid_scaling
    rtdose_dicom.PhotometricInterpretation = 'MONOCHROME2'
    rtdose_dicom.PixelRepresentation = 0  # unsigned
    rtdose_dicom.BitsAllocated = 16
    rtdose_dicom.BitsStored = 16
    rtdose_dicom.HighBit = 15
    rtdose_dicom.SamplesPerPixel = 1

    # Add dose data. 
    rtdose_dicom.PixelData = np.transpose(data).tobytes()     # Uses (z, y, x) spacing.

    # Set timestamps.
    dt = datetime.now()
    rtdose_dicom.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
    rtdose_dicom.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
    rtdose_dicom.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)

    return rtdose_dicom
