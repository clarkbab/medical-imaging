import cv2 as cv
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pydicom as dcm
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
import skimage as ski
from typing import *

from mymi.constants import DICOM_DATE_FORMAT, DICOM_TIME_FORMAT
from mymi import logging
from mymi.typing import *
from mymi.utils import *

CONTOUR_FORMATS = ['POINT', 'CLOSED_PLANAR']
CONTOUR_METHOD = 'SKIMAGE'
# CONTOUR_METHOD = 'OPENCV'

@dataclass
class LandmarksData:
    data: Tuple[float]
    name: str
    number: Optional[int] = None

@dataclass
class ROIData:
    colour: Colour
    data: np.ndarray
    name: str
    number: Optional[int] = None

class SOPClassUID:
    DETACHED_STUDY_MANAGEMENT = '1.2.840.10008.3.1.2.3.1'
    RTSTRUCT = '1.2.840.10008.5.1.4.1.1.481.3'
    RTSTRUCT_IMPLEMENTATION_CLASS = PYDICOM_IMPLEMENTATION_UID

class RtStructConverter:
    @classmethod
    def has_roi_data(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        name: str) -> np.ndarray:
        """
        returns: True if 'ContourData' is present for the ROI.
        args:
            rtstruct: the RTSTRUCT dicom.
            name: the ROI name.
            ref_cts: the reference CT dicoms.
        """
        # Load the contour data.
        roi_contours = rtstruct.ROIContourSequence
        roi_infos = rtstruct.StructureSetROISequence
        if len(roi_infos) != len(roi_contours):
            raise ValueError(f"Length of 'StructureSetROISequence' and 'ROIContourSequence' must be the same, got '{len(roi_infos)}' and '{len(roi_contours)}' respectively.")
        try:
            roi, _ = next(filter(lambda r: r[1].ROIName == name, zip(roi_contours, roi_infos)))
        except StopIteration:
            return False

        # Skip label if no contour sequence.
        contour_seq = getattr(roi, 'ContourSequence', None)
        if not contour_seq:
            return False

        return True

    @classmethod
    def get_roi_landmark(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        name: str) -> np.ndarray:
        """
        returns: an np.ndarray of mask data.
        returns: an (x, y, z) landmark location in patient coordinates [mm].
        args:
            rtstruct: the RTSTRUCT dicom.
            name: the ROI name.
        raises:
            ValueError: if name not found in ROIs, or no 'ContourSequence' data found.
        """
        # Load the contour data.
        roi_infos = rtstruct.StructureSetROISequence
        roi_contours = rtstruct.ROIContourSequence
        if len(roi_infos) != len(roi_contours):
            raise ValueError(f"Length of 'StructureSetROISequence' and 'ROIContourSequence' must be the same, got '{len(roi_infos)}' and '{len(roi_contours)}' respectively.")
        info_map = dict((info.ROIName, contour) for contour, info in zip(roi_contours, roi_infos))
        if name not in info_map:
            raise ValueError(f"RTSTRUCT doesn't contain ROI '{name}'.")
        roi_contour = info_map[name]

        # Skip label if no contour sequence.
        contour_seq = getattr(roi_contour, 'ContourSequence', None)
        if not contour_seq:
            raise ValueError(f"'ContourSequence' not found for ROI '{name}'.")

        # Filter contours without data.
        contour_seq = list(filter(lambda c: hasattr(c, 'ContourData'), contour_seq))

        # Should only be a single contour sequence for landmark data.
        if len(contour_seq) != 1:
            raise ValueError(f"Expected contour sequence of length 1 for landmark data. Got '{len(contour_seq)}'.")

        # Load landmark.
        contour = contour_seq[0]
        if contour.ContourGeometricType != 'POINT':
            raise ValueError(f"Expected contour type 'POINT' for landmark data. Got '{contour.ContourGeometricType}'.")
        landmark = np.array(contour.ContourData)

        return landmark

    @classmethod
    def get_regions_data(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        name: str,
        size: Size3D,
        spacing: Spacing3D,
        offset: Point3D) -> RegionsData:
        # Load the contour data.
        roi_infos = rtstruct.StructureSetROISequence
        roi_contours = rtstruct.ROIContourSequence
        if len(roi_infos) != len(roi_contours):
            raise ValueError(f"Length of 'StructureSetROISequence' and 'ROIContourSequence' must be the same, got '{len(roi_infos)}' and '{len(roi_contours)}' respectively.")
        info_map = dict((info.ROIName, contour) for contour, info in zip(roi_contours, roi_infos))
        if name not in info_map:
            raise ValueError(f"RTSTRUCT doesn't contain ROI '{name}'.")
        roi_contour = info_map[name]

        # Create label placeholder.
        data = np.zeros(shape=size, dtype=bool)

        # Skip label if no contour sequence.
        contour_seq = getattr(roi_contour, 'ContourSequence', None)
        if not contour_seq:
            raise ValueError(f"'ContourSequence' not found for ROI '{name}'.")

        # Filter contours without data.
        contour_seq = filter(lambda c: hasattr(c, 'ContourData'), contour_seq)

        # Sort contour sequence by z-axis.
        contour_seq = sorted(contour_seq, key=lambda c: c.ContourData[2])

        # Convert points into voxel data.
        for i, contour in enumerate(contour_seq):
            # Get contour data.
            contour_data = np.array(contour.ContourData)

            # This code handles 'CLOSED_PLANAR' and 'POINT' types.
            if not contour.ContourGeometricType in CONTOUR_FORMATS:
                raise ValueError(f"Expected one of '{CONTOUR_FORMATS}' ContourGeometricTypes, got '{contour.ContourGeometricType}' for contour '{i}', ROI '{name}'.")

            # Coords are stored in flat array.
            if contour_data.size % 3 != 0:
                raise ValueError(f"Size of 'contour_data' (array of points in 3D) should be divisible by 3.")
            points = np.array(contour_data).reshape(-1, 3)

            # Convert contour data to voxels.
            size_2D, spacing_2D, offset_2D = list(size)[:2], list(spacing)[:2], list(offset)[:2]
            slice_data = cls._get_mask_slice(points, size_2D, spacing_2D, offset_2D)

            # Get z index of slice.
            z_idx = int((points[0, 2] - offset[2]) / spacing[2])
            if z_idx > data.shape[2] - 1:
                # Happened with 'PMCC-COMP:PMCC_AI_GYN_011' - Kidney_L...
                continue

            # Write slice data to label, using XOR.
            data[:, :, z_idx][slice_data == True] = np.invert(data[:, :, z_idx][slice_data == True])

        return data

    @classmethod
    def _get_mask_slice(
        cls,
        points: np.ndarray,
        size: Size2D,
        spacing: Spacing2D,
        offset: Point2D) -> np.ndarray:
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
        x_indices = np.around(x_indices)                    # Round to avoid truncation errors.
        y_indices = np.around(y_indices)

        # Convert to 'cv2' format.
        indices = np.stack((y_indices, x_indices), axis=1)  # (y, x) as 'cv.fillPoly' expects rows, then columns.
        indices = indices.astype('int32')                   # 'cv.fillPoly' expects 'int32' input points.
        pts = [np.expand_dims(indices, axis=0)]

        # Get all voxels on the boundary and interior described by the indices.
        slice_data = np.zeros(shape=size, dtype='uint8')   # 'cv.fillPoly' expects to write to 'uint8' mask.
        cv.fillPoly(img=slice_data, pts=pts, color=1)
        slice_data = slice_data.astype(bool)

        return slice_data

    @classmethod
    def get_roi_names(
        cls,
        rtstruct: dcm.dataset.FileDataset) -> List[str]:
        """
        returns: a list of ROIs.
        args:
            rtstruct: the RTSTRUCT dicom.
        """
        # Load names.
        names = [i.ROIName for i in rtstruct.StructureSetROISequence]
        return names

    @classmethod
    def get_roi_info(
        cls,
        rtstruct: dcm.dataset.FileDataset) -> List[str]:
        """
        returns: a list of ROIs info.
        args:
            rtstruct: the RTSTRUCT dicom.
        """
        # Load info.
        info = dict((int(i.ROINumber), {
            'id': int(i.ROINumber),
            'name': i.ROIName,
        }) for i in rtstruct.StructureSetROISequence)
        return info

    @classmethod
    def create_rtstruct(
        cls,
        ref_cts: Sequence[dcm.dataset.FileDataset],
        info: Dict[str, str] = {}) -> dcm.dataset.FileDataset:
        """
        returns: an RTSTRUCT dicom.
        args:
            ref_cts: the reference CT dicoms.
        kwargs:
            info: a dictionary of extra info to add to RTSTRUCT.
                institution: the RTSTRUCT 'InstitutionName' field.
                label: the RTSTRUCT 'StructureSetLabel' field.
        """
        # Create metadata.
        metadata = cls.__create_metadata()

        # Create rtstruct.
        rtstruct = FileDataset('filename', {}, file_meta=metadata, preamble=b'\0' * 128)

        # Set empty sequences.
        rtstruct.StructureSetROISequence = dcm.sequence.Sequence()
        rtstruct.ROIContourSequence = dcm.sequence.Sequence()
        rtstruct.RTROIObservationsSequence = dcm.sequence.Sequence()

        # Add general info.
        cls.__add_general_info(rtstruct, info)

        # Add patient info.
        cls.__add_patient_info(rtstruct, ref_cts[0])

        # Add study/series info.
        cls.__add_study_and_series_info(rtstruct, ref_cts[0], info)

        # Add frame of reference.
        cls.__add_frames_of_reference(rtstruct, ref_cts)

        return rtstruct

    @classmethod
    def __create_metadata(cls) -> dcm.dataset.FileMetaDataset:
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
    def __add_general_info(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        info: Dict[str, str]) -> None:
        """
        effect: adds general info to RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            info: a dictionary of general info to add to RTSTRUCT.
        """
        # Set transfer syntax.
        rtstruct.is_little_endian = True
        rtstruct.is_implicit_VR = True

        # Set values from metadata.
        rtstruct.SOPClassUID = rtstruct.file_meta.MediaStorageSOPClassUID
        rtstruct.SOPInstanceUID = rtstruct.file_meta.MediaStorageSOPInstanceUID

        # Get date/time.
        dt = datetime.now()
        date = dt.strftime(DICOM_DATE_FORMAT)
        time = dt.strftime(DICOM_TIME_FORMAT)

        # Set other required fields.
        rtstruct.ContentDate = date
        rtstruct.ContentTime = time
        rtstruct.InstanceCreationDate = date
        rtstruct.InstanceCreationTime = time
        if 'institution' in info:
            rtstruct.InstitutionName = info['institution']
        rtstruct.Modality = 'rtstruct'
        rtstruct.SpecificCharacterSet = 'ISO_IR 100'
        rtstruct.StructureSetLabel = info['label'] if 'label' in info else 'rtstruct'
        rtstruct.StructureSetDate = date
        rtstruct.StructureSetTime = time

        # Set approval.
        rtstruct.ApprovalStatus = 'UNAPPROVED'

    @classmethod
    def __add_patient_info(
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
    def __add_study_and_series_info(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        ref_ct: dcm.dataset.FileDataset,
        info: Dict[str, str]) -> None:
        """
        effect: copies study/series info from the CT to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            ref_ct: the reference CT dicom.
            info: a dictionary of information.
        """
        # Get current datetime.
        dt = datetime.now()
 
        # Copy study info - same as reference CT.
        rtstruct.StudyDate = ref_ct.StudyDate
        rtstruct.StudyDescription = getattr(ref_ct, 'StudyDescription', '')
        rtstruct.StudyInstanceUID = ref_ct.StudyInstanceUID
        rtstruct.StudyID = ref_ct.StudyID
        rtstruct.StudyTime = ref_ct.StudyTime

        # Add series info.
        rtstruct.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
        rtstruct.SeriesInstanceUID = generate_uid()
        rtstruct.SeriesNumber = 0
        rtstruct.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)

    @classmethod
    def __add_frames_of_reference(
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
        # All CTs in the series should share the frame of reference. Read UID from
        # the first CT.
        frame = Dataset()
        # frame.FrameOfReferenceUID = generate_uid()
        frame.FrameOfReferenceUID = ref_cts[0].FrameOfReferenceUID

        # Add referenced study sequence.
        cls.__add_studies(frame, ref_cts)

        # Add frame of reference to RTSTRUCT.
        rtstruct.ReferencedFrameOfReferenceSequence = dcm.sequence.Sequence()
        rtstruct.ReferencedFrameOfReferenceSequence.append(frame)

    @classmethod
    def __add_studies(
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
        cls.__add_series(study, ref_cts)

        # Add study to the frame of reference. 
        frame.RTReferencedStudySequence = dcm.sequence.Sequence()
        frame.RTReferencedStudySequence.append(study) 

    @classmethod
    def __add_series(
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
        cls.__add_contour_images(series, ref_cts)

        # Add series to the study.
        study.RTReferencedSeriesSequence = dcm.sequence.Sequence()
        study.RTReferencedSeriesSequence.append(series)

    @classmethod
    def __add_contour_images(
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
            contour_image.ReferencedSOPClassUID = ct.file_meta.MediaStorageSOPClassUID
            contour_image.ReferencedSOPInstanceUID = ct.file_meta.MediaStorageSOPInstanceUID
            series.ContourImageSequence.append(contour_image)

    @classmethod
    def add_roi_landmark(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        name: str,
        landmark: Tuple[float],
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds landmark ROI to the RTSTRUCT dicom file.
        args:
            rtstruct: the RTSTRUCT dicom file.
            name: the landmark name.
            ref_cts: the reference CT dicoms.
        """
        # Perform checks.
        logging.info(name)
        logging.info(landmark)
        assert len(landmark) == 3

        # Get structure number.

        # Create ROI landmark.
        structure_set_landmark = Dataset()
        number = len(rtstruct.StructureSetROISequence) + 1
        structure_set_landmark.ROIGenerationAlgorithm = 'PMCC_AI'
        structure_set_landmark.ROIName = name
        structure_set_landmark.ROINumber = number
        structure_set_landmark.ReferencedFrameOfReferenceUID = rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
        rtstruct.StructureSetROISequence.append(structure_set_landmark)

        # Add the contour data.
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        roi_contour.ROIDisplayColor = list(to_rgb_255((1, 1, 0)))   # Yellow.
        roi_contour.ContourSequence = dcm.sequence.Sequence()
        contour = Dataset()
        contour.NumberOfContourPoints = 1
        contour.ContourGeometricType = 'POINT'
        contour.ContourData = list(landmark)
        zs = np.array([float(c.ImagePositionPatient[2]) for c in ref_cts])
        # Get the closest CT to the landmark.
        ref_ct = ref_cts[np.abs(zs - landmark[2]).argmin()]
        ref_image = Dataset()
        ref_image.ReferencedSOPClassUID = ref_ct.file_meta.MediaStorageSOPClassUID
        ref_image.ReferencedSOPInstanceUID = ref_ct.file_meta.MediaStorageSOPInstanceUID
        contour.ContourImageSequence = dcm.sequence.Sequence()
        contour.ContourImageSequence.append(ref_image)
        roi_contour.ContourSequence.append(contour)
        rtstruct.ROIContourSequence.append(roi_contour)

    @classmethod
    def add_roi_contour(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        roi_data: ROIData,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds ROI data to the RTSTRUCT dicom file.
        args:
            rtstruct: the RTSTRUCT dicom file.
            name: the ROI name.
            roi_data: the ROIData instance.
            ref_cts: the reference CT dicoms.
        """
        # Perform checks.
        assert roi_data.data.dtype == bool
        assert roi_data.data.ndim == 3
        # assert roi_data.data.sum() != 0       # Some network predictions are empty.

        # Add ROI number.
        if roi_data.number is None:
            roi_data.number = len(rtstruct.StructureSetROISequence) + 1

        # Add ROI contours.
        cls.__add_roi_contours(rtstruct, roi_data, ref_cts)

        # Add structure set ROIs.
        cls.__add_structure_set_rois(rtstruct, roi_data)

        # Add RT ROI observations.
        cls.__add_rt_roi_observations(rtstruct, roi_data)

    @classmethod
    def __add_roi_contours(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        roi_data: ROIData,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds ROI contours sequence to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            roi_data: ROI information.
            ref_cts: the reference CT dicoms.
        """
        # Create ROI contour.
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = roi_data.colour
        roi_contour.ReferencedROINumber = str(roi_data.number)

        # Add contour sequence.
        cls.__add_contours(roi_contour, roi_data.data, ref_cts)

        # Append ROI contour.
        rtstruct.ROIContourSequence.append(roi_contour)

    @classmethod
    def __add_contours(
        cls,
        roi_contour: dcm.dataset.Dataset,
        data: np.ndarray,
        ref_cts: Sequence[dcm.dataset.FileDataset]) -> None:
        """
        effect: adds contours to the ROI contour.
        args:
            roi_contour: the ROI contour.
            data: the contour data array.
            ref_cts: the reference CT dicoms.
        """
        # Create contour sequence.
        roi_contour.ContourSequence = dcm.sequence.Sequence()

        for i, ct in enumerate(ref_cts):
            # Get data slice.
            slice_data = data[:, :, i]
            
            # Skip empty slices.
            if slice_data.sum() == 0:
                continue

            # Add contour.
            cls.__add_slice_contours(roi_contour, slice_data, ct, i)

    @classmethod
    def __add_slice_contours(
        cls,
        roi_contour: dcm.dataset.Dataset,
        slice_data: np.ndarray,
        ref_ct: dcm.dataset.FileDataset,
        slice_idx: int) -> None:
        """
        effect: adds a contour to the ROI contour.
        args:
            roi_contour: the ROI contour.
            slice_data: the slice data.
            ref_ct: the reference CT dicom.
        """
        # Get contour coordinates.
        slice_data = slice_data.astype('uint8')
        if CONTOUR_METHOD == 'OPENCV':
            # contours_coords, _ = cv.findContours(slice_data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            # 'CHAIN_APPROX_SIMPLE' tries to replace straight line boundaries with two end points, instead
            # of using many points along the line - however it was producing only two points for some small
            # structures which Velocity doesn't like.
            contours_coords, _ = cv.findContours(slice_data, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

            # Process results.
            # Each slice can have multiple contours - separate foreground regions in the image.
            for i, c in enumerate(contours_coords):
                # OpenCV adds intermediate dimension - for some reason?
                c = c.squeeze(1)

                # Change in v4.11?
                # # OpenCV returns (y, x) points, so flip.
                c = np.flip(c, axis=1)
                contours_coords[i] = c

        elif CONTOUR_METHOD == 'SKIMAGE':
            contours_coords = ski.measure.find_contours(slice_data)
            # Skimage needs no post-processing, as it returns (x, y) along the same
            # axes as the passed 'slice_data'. Also no strange intermediate dimensions.
        else:
            raise ValueError(f"CONTOUR_METHOD={CONTOUR_METHOD} not recognised.")

        contours_coords = list(contours_coords)

        # Velocity has an issue with loading contours that contain less than 3 points.
        for i, c in enumerate(contours_coords):
            if len(c) < 3:
                raise ValueError(f"Contour {i} of slice {slice_idx} contains only 3 points: {contours_coords}. Velocity will not like this.")

        # 'contours_coords' is a list of contour coordinates, i.e. multiple contours are possible per slice.
        for contour_coords in contours_coords:
            # Convert to numpy array.
            contour_coords = np.array(contour_coords)

            # Translate to physical space.
            offset = ref_ct.ImagePositionPatient
            spacing = ref_ct.PixelSpacing
            contour_coords = spacing * contour_coords + offset[:-1]

            # Format for DICOM.
            contour_coords = cls._format_coordinates(contour_coords, ref_ct)

            # Create contour.
            contour = Dataset()
            contour.ContourData = contour_coords
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = len(contour_coords) / 3

            # Add contour images.
            cls.__add_roi_contour_images(contour, ref_ct)

            # Append contour to ROI contour.
            roi_contour.ContourSequence.append(contour)

    @classmethod
    def _format_coordinates(
        cls,
        coords: np.ndarray,
        ref_ct: dcm.dataset.FileDataset) -> List[float]:
        """
        returns: the coordinates formated to DICOM standard.
        args:
            coords: the coordinates in physical space.
            ref_ct: the reference CT dicom.
        """
        # Add z-index.
        z_indices = np.ones((len(coords), 1)) * ref_ct.ImagePositionPatient[2]
        coords = np.concatenate((coords, z_indices), axis=1)

        # Flatten the array.
        coords = coords.flatten()

        # Convert to list.
        coords = list(coords)

        return coords 

    @classmethod
    def __add_roi_contour_images(
        cls,
        contour: dcm.dataset.Dataset,
        ref_ct: dcm.dataset.FileDataset) -> None:
        """
        effect: adds reference images to the contour.
        args:
            contour: the contour.
            ref_ct: the reference CT dicom.
        """
        # Create contour image.
        image = Dataset()
        image.ReferencedSOPClassUID = ref_ct.file_meta.MediaStorageSOPClassUID
        image.ReferencedSOPInstanceUID = ref_ct.file_meta.MediaStorageSOPInstanceUID

        # Append to contour.
        contour.ContourImageSequence = dcm.sequence.Sequence()
        contour.ContourImageSequence.append(image)

    @classmethod
    def __add_structure_set_rois(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        roi_data: ROIData) -> None:
        """
        effect: adds structure set ROIs to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
        """
        # Create structure set roi.
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = roi_data.number
        structure_set_roi.ReferencedFrameOfReferenceUID = rtstruct.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
        structure_set_roi.ROIName = roi_data.name
        structure_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'

        # Add to RTSTRUCT dicom.
        rtstruct.StructureSetROISequence.append(structure_set_roi)

    @classmethod
    def __add_rt_roi_observations(
        cls,
        rtstruct: dcm.dataset.FileDataset,
        roi_data: ROIData) -> None:
        """
        effect: add RT ROI observations to the RTSTRUCT dicom.
        args:
            rtstruct: the RTSTRUCT dicom.
            roi_data: the ROIData instance.
        """
        # Create RT ROI observation.
        observation = Dataset()
        observation.ObservationNumber = roi_data.number
        observation.ReferencedROINumber = roi_data.number
        observation.RTROIInterpretedType = ''
        observation.ROIInterpreter = ''

        # Add to RTSTRUCT.
        rtstruct.RTROIObservationsSequence.append(observation)
