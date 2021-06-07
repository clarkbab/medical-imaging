import cv2 as cv
from datetime import datetime
import numpy as np
import pydicom as dcm
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
from typing import Dict, Sequence

from mymi import types

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
        patient_id: types.PatientID,
        rois: Dict[str, np.ndarray],
        ref_ct: dcm.dataset.FileDataset) -> dcm.dataset.FileDataset:
        """
        returns: an RTSTRUCT dicom.
        args:
            patient_id: the patient ID.
            rois: a dict with roi name keys and mask data values.
            ref_ct: the reference CT dicom.
        """
        # Create metadata.
        metadata = cls._create_metadata()

        # Create rtstruct.
        rtstruct = FileDataset('filename', {}, file_meta=metadata, preamble=b'\0' * 128)

        # Add general data.
        cls._add_general_info(rtstruct)

        # Add patient data.
        cls._add_patient_info(rtstruct, patient_id)

        # Add study/series data.
        cls._add_study_and_series_info(rtstruct, ref_ct)

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
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
        file_meta.MediaStorageSOPInstanceUID = cls._generate_uid()
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID

        return file_meta

    @classmethod
    def _generate_uid(cls) -> str:
        """
        returns: a new RTSTRUCT dicom UID.
        """
        pass

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
        rtstruct: dcm.dataset.FileDataset,
        id: str) -> None:
        """
        effect: adds patient info to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            id: the patient ID.
        """
        # Add patient info.
        pat = dataset.patient(id)
        rtstruct.PatientName = getattr(name, 'PatientName', '')
        rtstruct.PatientID = getattr(id, 'PatientID', '')
        rtstruct.PatientBirthDate = getattr(birth_date, 'PatientBirthDate', '')
        rtstruct.PatientSex = getattr(sex, 'PatientSex', '')
        rtstruct.PatientAge = getattr(age, 'PatientAge', '')
        rtstruct.PatientSize = getattr(size, 'PatientSize', '')
        rtstruct.PatientWeight = getattr(weight, 'PatientWeight', '')

    @classmethod
    def _add_study_and_series_info(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        ref_ct: dcm.dataset.FileDataset) -> None:
        """
        effect: copies study/series info from the CT to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            ref_ct: the reference CT dicom.
        """
        # Copy information.
        ds.StudyDate = reference_ds.StudyDate
        ds.SeriesDate = getattr(reference_ds, 'SeriesDate', '')
        ds.StudyTime = reference_ds.StudyTime
        ds.SeriesTime = getattr(reference_ds, 'SeriesTime', '')
        ds.StudyDescription = getattr(reference_ds, 'StudyDescription', '')
        ds.SeriesDescription = getattr(reference_ds, 'SeriesDescription', '')
        ds.StudyInstanceUID = reference_ds.StudyInstanceUID
        ds.SeriesInstanceUID = generate_uid() # TODO: find out if random generation is ok
        ds.StudyID = reference_ds.StudyID
        ds.SeriesNumber = "1" # TODO: find out if we can just use 1 (Should be fine since its a new series)

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