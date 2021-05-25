import cv2 as cv
import numpy as np
import pydicom as dcm
from typing import *

class RTStructConverter:
    @classmethod
    def get_roi_mask(
        cls,
        name: str,
        offset: Tuple[float, float, float],
        rtstruct: dcm.dataset.FileDataset,
        shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        returns: an np.ndarray of mask data.
        args:
            name: the ROI name.
            offset: the (0, 0, 0) voxel offset in physical space.
            rtstruct: the RTSTRUCT dicom.
            shape: the mask shape.
            spacing: the (x, y, z) voxel spacing in mm.
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
        data = np.zeros(shape=shape, dtype=bool)

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
            slice_data = cls._get_mask_slice(offset[:-1], points, shape[:-1], spacing[:-1])

            # Write slice data to label, using XOR.
            data[:, :, z_idx][slice_data == True] = np.invert(data[:, :, z_idx][slice_data == True])

        return data

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
    def _get_mask_slice(
        cls,
        offset: Tuple[float, float],
        points: np.ndarray,
        shape: Tuple[int, int],
        spacing: Tuple[float, float]) -> np.ndarray:
        """
        returns: a boolean np.ndarray containing the mask for the slice.
        args:
            offset: the (0, 0) pixel offset in physical space.
            points: the (n x 3) np.ndarray of contour points in physical space.
            shape: the resulting mask shape.
            spacing: the (x, y) pixel spacing in mm.
        """
        # Convert from physical coordinates to array indices.
        x_indices = (points[:, 0] - offset[0]) / spacing[0]
        y_indices = (points[:, 1] - offset[1]) / spacing[1]

        # Round before typecasting to avoid truncation.
        indices = np.stack((y_indices, x_indices), axis=1)  # (y, x) as 'cv.fillPoly' expects rows, then columns.
        indices = np.around(indices)    # Round to avoid truncation errors.
        indices = indices.astype('int32')   # 'cv.fillPoly' expects 'int32' input points.

        # Get all voxels on the boundary and interior described by the indices.
        slice_data = np.zeros(shape=shape, dtype='uint8')   # 'cv.fillPoly' expects to write to 'uint8' mask.
        pts = [np.expand_dims(indices, axis=0)]
        cv.fillPoly(img=slice_data, pts=pts, color=1)
        slice_data = slice_data.astype(bool)

        return slice_data

    @classmethod
    def create_rtstruct(
        cls,
        labels: dict) -> dcm.dataset.FileDataset:
        """
        returns: an RTSTRUCT dicom.
        args:
            labels: a dict with label name keys and mask data values.
        """
        # Create pydicom RTSTRUCT file.

    @classmethod
    def _get_contours(
        cls,
        mask: np.ndarray) -> 
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