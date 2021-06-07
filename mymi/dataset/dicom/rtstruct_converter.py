import cv2 as cv
from datetime import datetime
import numpy as np
import pydicom as dcm
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
from typing import Dict, Sequence

from mymi import types

class SOPClassUID:
    DETACHED_STUDY_MANAGEMENT = '1.2.840.10008.3.1.2.3.1'
    RTSTRUCT = '1.2.840.10008.5.1.4.1.1.481.3'
    RTSTRUCT_IMPLEMENTATION_CLASS = PYDICOM_IMPLEMENTATION_UID

class RTStructConverter:
    @classmethod
    def get_roi_mask(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        name: str,
        size: types.Size3D,
        spacing: types.Spacing3D,
        offset: types.Point3D) -> np.ndarray:
        """
        returns: an np.ndarray of mask data.
        args:
            rtstruct: the RTSTRUCT dicom.
            name: the ROI name.
            size: the mask size.
            spacing: the (x, y, z) voxel spacing in mm.
            offset: the (0, 0, 0) voxel offset in physical space.
        raises:
            ValueError: if name not found in ROIs, or no 'ContourSequence' data found.
        """
        # Load the contour data.
        rois = rtstruct.ROIContourSequence
        roi_infos = rtstruct.StructureSetROISequence
        try:
            roi, _ = next(filter(lambda r: r[1].ROIName == name, zip(rois, roi_infos)))
        except StopIteration:
            raise ValueError(f"RTSTRUCT doesn't contain ROI '{name}'.")

        # Create label placeholder.
        data = np.zeros(shape=size, dtype=bool)

        # Skip label if no contour sequence.
        contour_seq = getattr(roi, 'ContourSequence', None)
        if not contour_seq:
            raise ValueError(f"'ContourSequence' not found for ROI '{name}'.")

        # Convert points into voxel data.
        for i, contour in enumerate(contour_seq):
            # Get contour data.
            contour_data = contour.ContourData
            if contour.ContourGeometricType != 'CLOSED_PLANAR':
                raise ValueError(f"Expected contour type 'CLOSED_PLANAR', got '{contour.ContourGeometricType}'.")

            # Coords are stored in flat array.
            points = np.array(contour_data).reshape(-1, 3)

            # Get z_idx of slice.
            z_idx = int((points[0, 2] - offset[2]) / spacing[2])

            # Convert contour data to voxels.
            slice_data = cls._get_mask_slice(points, size[:-1], spacing[:-1], offset[:-1])

            # Write slice data to label, using XOR.
            data[:, :, z_idx][slice_data == True] = np.invert(data[:, :, z_idx][slice_data == True])

        return data

    @classmethod
    def get_roi_names(
        cls,
        rtstruct: dcm.dataset.FileDataset) -> Sequence[str]:
        """
        returns: a list of ROIs.
        args:
            rtstruct: the RTSTRUCT dicom.
        """
        # Load names.
        names = [i.ROIName for i in rtstruct.StructureSetROISequence]
        return names

    @classmethod
    def _get_mask_slice(
        cls,
        points: np.ndarray,
        size: types.Size2D,
        spacing: types.Spacing2D,
        offset: types.Point2D) -> np.ndarray:
        """
        returns: the boolean array mask for the slice.
        args:
            points: the (n x 3) np.ndarray of contour points in physical space.
            size: the resulting mask size.
            spacing: the (x, y) pixel spacing in mm.
            offset: the (0, 0) pixel offset in physical space.
        """
        # Convert from physical coordinates to array indices.
        x_indices = (points[:, 0] - offset[0]) / spacing[0]
        y_indices = (points[:, 1] - offset[1]) / spacing[1]

        # Round before typecasting to avoid truncation.
        indices = np.stack((y_indices, x_indices), axis=1)  # (y, x) as 'cv.fillPoly' expects rows, then columns.
        indices = np.around(indices)    # Round to avoid truncation errors.
        indices = indices.astype('int32')   # 'cv.fillPoly' expects 'int32' input points.

        # Get all voxels on the boundary and interior described by the indices.
        slice_data = np.zeros(shape=size, dtype='uint8')   # 'cv.fillPoly' expects to write to 'uint8' mask.
        pts = [np.expand_dims(indices, axis=0)]
        cv.fillPoly(img=slice_data, pts=pts, color=1)
        slice_data = slice_data.astype(bool)

        return slice_data

    @classmethod
    def create_rtstruct(
        cls,
        rois: Dict[str, np.ndarray],
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> dcm.dataset.FileDataset:
        """
        returns: an RTSTRUCT dicom.
        args:
            rois: a dict with roi name keys and mask data values.
            ref_cts: the reference CT dicoms.
        """
        # Create metadata.
        metadata = cls._create_metadata()

        # Create rtstruct.
        rtstruct = FileDataset('filename', {}, file_meta=metadata, preamble=b'\0' * 128)
        # rtstruct.StructureSetROISequence = dcm.sequence.Sequence()
        # rtstruct.ROIContourSequence = dcm.sequence.Sequence()
        # rtstruct.RTROIObservationsSequence = dcm.sequence.Sequence()

        # Add general info.
        cls._add_general_info(rtstruct)

        # Add patient info.
        cls._add_patient_info(rtstruct, ref_cts[0])

        # Add study/series info.
        cls._add_study_and_series_info(rtstruct, ref_cts[0])

        # Add frame of reference.
        cls._add_frame_of_reference(rtstruct, ref_cts)

        # Add ROI data. 
        cls._add_roi_data(rtstruct, rois)

        return rtstruct

    @classmethod
    def _create_metadata(cls) -> dcm.dataset.FileMetaDataset:
        """
        returns: a dicom FileMetaDataset containing RTSTRUCT metadata.
        """
        # Create metadata.
        file_meta = FileMetaDataset()
        file_meta.FileMetaInformationGroupLength = 204
        file_meta.FileMetaInformationVersion = b'\x00\x01'
        file_meta.MediaStorageSOPClassUID = SOPClassUID.RTSTRUCT
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.ImplementationClassUID = SOPClassUID.RTSTRUCT_IMPLEMENTATION_CLASS

        return file_meta

    @classmethod
    def _add_general_info(
        cls,
        rtstruct: dcm.dataset.FileDataset) -> None:
        """
        effect: adds general info to RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
        """
        # Set transfer syntax.
        rtstruct.is_little_endian = True
        rtstruct.is_implicit_VR = True

        # Set values from metadata.
        rtstruct.SOPClassUID = rtstruct.file_meta.MediaStorageSOPClassUID
        rtstruct.SOPInstanceUID = rtstruct.file_meta.MediaStorageSOPInstanceUID

        # Get date/time.
        dt = datetime.now()
        date = dt.strftime('%Y%m%d')
        time = dt.strftime('%H%M%S.%f')

        # Set other required fields.
        rtstruct.ContentDate = date
        rtstruct.ContentTime = time
        rtstruct.InstanceCreationDate = date
        rtstruct.InstanceCreationTime = time
        rtstruct.InstitutionName = 'PMCC'
        rtstruct.Manufacturer = 'PMCC'
        rtstruct.ManufacturerModelName = 'PMCC-Seg'
        rtstruct.Modality = 'RTSTRUCT'
        rtstruct.SpecificCharacterSet = 'ISO_IR 100'
        rtstruct.StructureSetLabel = 'RTSTRUCT'
        rtstruct.StructureSetDate = date
        rtstruct.StructureSetTime = time

        # Set approval.
        rtstruct.ApprovalStatus = 'UNAPPROVED'

    @classmethod
    def _add_patient_info(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        ref_ct: dcm.dataset.FileDataset) -> None:
        """
        effect: adds patient info to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            ref_ct: the reference CT dicom.
        """
        # Add patient info.
        rtstruct.PatientAge = getattr(ref_ct, 'PatientAge', '')
        rtstruct.PatientBirthDate = getattr(ref_ct, 'PatientBirthDate', '')
        rtstruct.PatientID = getattr(ref_ct, 'PatientID', '')
        rtstruct.PatientName = getattr(ref_ct, 'PatientName', '')
        rtstruct.PatientSex = getattr(ref_ct, 'PatientSex', '')
        rtstruct.PatientSize = getattr(ref_ct, 'PatientSize', '')
        rtstruct.PatientWeight = getattr(ref_ct, 'PatientWeight', '')

    @classmethod
    def _add_study_and_series_info(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        ref_ct: dcm.dataset.FileDataset) -> None:
        """
        effect: copies study/series info from the CT to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            ref_ct: the reference CT dicom.  """
        # Copy information.
        rtstruct.SeriesDate = getattr(ref_ct, 'SeriesDate', '')
        rtstruct.SeriesDescription = getattr(ref_ct, 'SeriesDescription', '')
        rtstruct.SeriesInstanceUID = generate_uid()
        rtstruct.SeriesNumber = 1
        rtstruct.SeriesTime = getattr(ref_ct, 'SeriesTime', '')
        rtstruct.StudyDate = ref_ct.StudyDate
        rtstruct.StudyDescription = getattr(ref_ct, 'StudyDescription', '')
        rtstruct.StudyInstanceUID = ref_ct.StudyInstanceUID
        rtstruct.StudyID = ref_ct.StudyID
        rtstruct.StudyTime = ref_ct.StudyTime

    @classmethod
    def _add_frame_of_reference(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds frame of reference to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            ref_cts: the reference CT dicoms.
        """
        # Create frame of reference.
        frame = Dataset()
        frame.FrameOfReferenceUID = generate_uid()

        # Add referenced study sequence.
        cls._add_study(frame, ref_cts)

        # Add frame of reference to RTSTRUCT.
        rtstruct.ReferencedFrameOfReferenceSequence = dcm.sequence.Sequence()
        rtstruct.ReferencedFrameOfReferenceSequence.append(frame)

    @classmethod
    def _add_study(
        cls,
        frame: dcm.dataset.Dataset,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds referenced CT study.
        args:
            frame: the frame of reference.
            ref_cts: the reference CT dicoms.
        """
        # Create study.
        study = Dataset()
        study.ReferencedSOPClassUID = SOPClassUID.DETACHED_STUDY_MANAGEMENT
        study.ReferencedSOPInstanceUID = ref_cts[0].StudyInstanceUID

        # Add contour image sequence.
        cls._add_series(study, ref_cts)

        # Add study to the frame of reference. 
        frame.RTReferencedStudySequence = dcm.sequence.Sequence()
        frame.RTReferencedStudySequence.append(study) 

    @classmethod
    def _add_series(
        cls,
        study: dcm.dataset.Dataset,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds referenced CT series.
        args:
            study: the CT study.
            ref_cts: the referenced CT dicoms.
        """
        # Create series.
        series = Dataset()
        series.SeriesInstanceUID = ref_cts[0].SeriesInstanceUID

        # Add contour image sequence.
        cls._add_contour_image_sequence(series, ref_cts)

        # Add series to the study.
        study.RTReferencedSeriesSequence = dcm.sequence.Sequence()
        study.RTReferencedSeriesSequence.append(series)

    @classmethod
    def _add_contour_image_series(
        cls,
        series: dcm.dataset.Dataset,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds contour images to CT series.
        args:
            series: the reference CT series.
            ref_cts: the referenced CT dicoms.
        """
        # Initialise contour image sequence.
        series.ContourImageSequence = dcm.sequence.Sequence()
        
        # Append contour images.
        for ct in ref_cts:
            contour_image = dcm.dataset.Dataset()
            contour_image.ReferencedSOPClassUID = series.file_meta.MediaStorageSOPClassUID
            contour_image.ReferencedSOPInstanceUID = series.file_meta.MediaStorageSOPInstanceUID
            series.ContourImageSequence.append(contour_image)

    @classmethod
    def _add_roi_data(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        rois: Dict[str, np.ndarray]) -> None:
        """
        effect: adds ROI data to the RTSTRUCT dicom file.
        args:
            rtstruct: the RTSTRUCT dicom file.
            rois: a dict of ROI data.
        """
        pass

    @classmethod
    def _get_contours(
        cls,
        mask: np.ndarray) -> np.ndarray:
        """
        returns: an array of contour vertices.
        args:
            mask: the binary input mask.
        """
        # Get contours.
        mask = mask.astype('uint8')
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Contours is a list of ndarrays containing the (x, y) coordinates of the vertices for each contour.
        # Hierarchy represented as [next, previous, first_child, parent]. Next/previous relate to contours
        # in the same level (-1 means no next/previous).
        # Without hierarchy we get:
        # [[1, -1, -1, -1],
        #  [2, 0, -1, -1],
        #  [3, 1, -1, -1] ... ]
        # I don't think we need to worry about 'hierarchy' unless we're creating a pinhole. Most
        # DICOM viewers will handle the contour hierarchy for us.
