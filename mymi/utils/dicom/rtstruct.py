import cv2 as cv
from datetime import datetime
import pydicom as dcm
import seaborn as sns
import skimage as ski
from typing import *

from mymi.constants import *
from mymi.typing import *
from mymi.utils.affine import affine_origin, affine_spacing
from mymi.utils.args import arg_to_list
from mymi.utils.dicom import add_landmarks_data, add_regions_data, from_ct_dicom, from_rtstruct_dicom, get_contour_coords, list_rtstruct_regions, to_rtstruct_dicom

CONTOUR_FORMATS = ['POINT', 'CLOSED_PLANAR']
CONTOUR_METHOD = 'SKIMAGE'
# CONTOUR_METHOD = 'OPENCV'

def from_rtstruct_dicom(
    rtstruct: FilePath | RtStructDicom,
    affine: Affine | None = None,
    cts: DirPath | List[CtDicom] = None,
    origin: Point3D | None = None,
    regions: RegionID | List[RegionID] | Literal['all'] = 'all',
    size: Size3D | None = None,
    ) -> LabelVolumeBatch:
    if isinstance(rtstruct, str):
        rtstruct = dcm.dcmread(rtstruct, force=False)
    if cts is None:
        assert size is not None
        assert affine is not None
    else:
        # Set shape and affine from CT if provided.
        ct_data, affine = from_ct_dicom(cts)
        size = ct_data.shape
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    regions = arg_to_list(regions, str, exceptions='all')

    # Load the contour data.
    roi_infos = rtstruct.StructureSetROISequence
    roi_contours = rtstruct.ROIContourSequence
    if len(roi_infos) != len(roi_contours):
        raise ValueError(f"Length of 'StructureSetROISequence' and 'ROIContourSequence' must be the same, got '{len(roi_infos)}' and '{len(roi_contours)}' respectively.")
    info_map = dict((info.ROIName, contour) for contour, info in zip(roi_contours, roi_infos))
    if regions == 'all':
        regions = list(info_map.keys())

    # Add regions data.
    data = np.zeros(shape=(len(regions), *size), dtype=np.bool_)
    for i, r in enumerate(regions):
        if r not in info_map:
            raise ValueError(f"RTSTRUCT doesn't contain ROI '{r}'.")
        roi_contour = info_map[r]

        # Skip label if no contour sequence.
        contour_seq = getattr(roi_contour, 'ContourSequence', None)
        if not contour_seq:
            raise ValueError(f"'ContourSequence' not found for ROI '{r}'.")

        # Filter contours without data.
        contour_seq = filter(lambda c: hasattr(c, 'ContourData'), contour_seq)

        # Sort contour sequence by z-axis.
        contour_seq = sorted(contour_seq, key=lambda c: c.ContourData[2])

        # Convert points into voxel data.
        for j, contour in enumerate(contour_seq):
            # Get contour data.
            contour_data = np.array(contour.ContourData)

            # This code handles 'CLOSED_PLANAR' and 'POINT' types.
            if not contour.ContourGeometricType in CONTOUR_FORMATS:
                raise ValueError(f"Expected one of '{CONTOUR_FORMATS}' ContourGeometricTypes, got '{contour.ContourGeometricType}' for contour '{j}', ROI '{r}'.")

            # Coords are stored in flat array.
            if contour_data.size % 3 != 0:
                raise ValueError(f"Size of 'contour_data' (array of points in 3D) should be divisible by 3.")
            points = np.array(contour_data).reshape(-1, 3)

            # Convert from physical coordinates to array indices.
            x_indices = (points[:, 0] - origin[0]) / spacing[0]
            y_indices = (points[:, 1] - origin[1]) / spacing[1]
            x_indices = np.around(x_indices)                    # Round to avoid truncation errors.
            y_indices = np.around(y_indices)

            # Convert to 'cv2' format.
            indices = np.stack((y_indices, x_indices), axis=1)  # (y, x) as 'cv.fillPoly' expects rows, then columns.
            indices = indices.astype('int32')                   # 'cv.fillPoly' expects 'int32' input points.
            pts = [np.expand_dims(indices, axis=0)]

            # Get all voxels on the boundary and interior described by the indices.
            slice_data = np.zeros(shape=size[:2], dtype='uint8')   # 'cv.fillPoly' expects to write to 'uint8' mask.
            cv.fillPoly(img=slice_data, pts=pts, color=1)
            slice_data = slice_data.astype(np.bool_)

            # Get z index of slice.
            z_idx = int((points[0, 2] - origin[2]) / spacing[2])
            if z_idx > size[2] - 1:
                # Happened with 'PMCC-COMP:PMCC_AI_GYN_011' - Kidney_L...
                continue

            # Write slice data to label, using XOR.
            data[i, :, :, z_idx][slice_data == True] = np.invert(data[i, :, :, z_idx][slice_data == True])

    return data

def list_rtstruct_regions(
    rtstruct: FilePath | RtStructDicom,
) -> List[Region]:
    if isinstance(rtstruct, str):
        rtstruct = dcm.dcmread(rtstruct, force=False)
    roi_infos = rtstruct.StructureSetROISequence
    regions = list(info.ROIName for info in roi_infos)
    return regions

def to_rtstruct_dicom(
    ref_cts: List[CtDicom],
    landmarks_data: Optional[LandmarksFrame] = None,
    landmark_prefix: str = '',
    regions_data: Optional[RegionArrays] = None,
    series: Optional[SeriesID] = None) -> RtStructDicom:
    if landmarks_data is None and regions_data is None:
        raise ValueError("At least one of 'landmarks_data' or 'regions_data' must be provided.")

    # Create metadata.
    file_meta = dcm.dataset.FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 204
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.ImplementationClassUID = dcm.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.MediaStorageSOPClassUID = dcm.uid.RTStructureSetStorage
    file_meta.MediaStorageSOPInstanceUID = dcm.uid.generate_uid()
    file_meta.TransferSyntaxUID = dcm.uid.ImplicitVRLittleEndian

    # Create RTSTRUCT.
    rtstruct_dicom = dcm.dataset.FileDataset('filename', {}, file_meta=file_meta, preamble=b'\0' * 128)
    rtstruct_dicom.ApprovalStatus = 'UNAPPROVED'
    rtstruct_dicom.InstitutionName = 'PMCC'
    rtstruct_dicom.Modality = 'RTSTRUCT'
    rtstruct_dicom.Manufacturer = 'PMCC'
    rtstruct_dicom.SOPClassUID = file_meta.MediaStorageSOPClassUID
    rtstruct_dicom.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    rtstruct_dicom.SpecificCharacterSet = 'ISO_IR 100'
    rtstruct_dicom.StructureSetLabel = 'RTSTRUCT'

    # Add patient info.
    rtstruct_dicom.PatientAge = getattr(ref_cts[0], 'PatientAge', '')
    rtstruct_dicom.PatientBirthDate = getattr(ref_cts[0], 'PatientBirthDate', '')
    rtstruct_dicom.PatientID = getattr(ref_cts[0], 'PatientID', '')
    rtstruct_dicom.PatientName = getattr(ref_cts[0], 'PatientName', '')
    rtstruct_dicom.PatientSex = getattr(ref_cts[0], 'PatientSex', '')
    rtstruct_dicom.PatientSize = getattr(ref_cts[0], 'PatientSize', '')
    rtstruct_dicom.PatientWeight = getattr(ref_cts[0], 'PatientWeight', '')

    # Add study info - from ref CT.
    rtstruct_dicom.StudyDate = ref_cts[0].StudyDate
    rtstruct_dicom.StudyDescription = getattr(ref_cts[0], 'StudyDescription', '')
    rtstruct_dicom.StudyID = ref_cts[0].StudyID
    rtstruct_dicom.StudyInstanceUID = ref_cts[0].StudyInstanceUID
    rtstruct_dicom.StudyTime = ref_cts[0].StudyTime

    # Add series info.
    series = f'RTSTRUCT ({ref_cts[0].StudyID})' if series is None else series
    rtstruct_dicom.SeriesDescription = series,
    rtstruct_dicom.SeriesInstanceUID = dcm.uid.generate_uid()
    rtstruct_dicom.SeriesNumber = 1

    # Add frame of reference.
    ref_ct_series = dcm.dataset.Dataset()
    ref_ct_series.SeriesInstanceUID = ref_cts[0].SeriesInstanceUID
    ref_study = dcm.dataset.Dataset()
    ref_study.ReferencedSOPClassUID = ref_cts[0].SOPClassUID
    ref_study.ReferencedSOPInstanceUID = ref_cts[0].StudyInstanceUID
    ref_study.RTReferencedSeriesSequence = [ref_ct_series]
    ref_frame = dcm.dataset.Dataset()
    ref_frame.FrameOfReferenceUID = ref_cts[0].FrameOfReferenceUID
    ref_frame.RTReferencedStudySequence = [ref_study]
    rtstruct_dicom.ReferencedFrameOfReferenceSequence = [ref_frame]

    # Add references to the CT images.
    ref_ct_series.ContourImageSequence = []
    for c in ref_cts:
        ref_image = dcm.dataset.Dataset()
        ref_image.ReferencedSOPClassUID = c.file_meta.MediaStorageSOPClassUID
        ref_image.ReferencedSOPInstanceUID = c.file_meta.MediaStorageSOPInstanceUID
        ref_ct_series.ContourImageSequence.append(ref_image)

    # Add contours.
    rtstruct_dicom.ROIContourSequence = []
    if regions_data is not None:
        rtstruct_dicom = add_regions_data(rtstruct_dicom, regions_data, ref_cts)

    # Add landmarks.
    if landmarks_data is not None:
        rtstruct_dicom = add_landmarks_data(rtstruct_dicom, landmarks_data, ref_cts, landmark_prefix=landmark_prefix)

    # Set timestamps.
    dt = datetime.now()
    rtstruct_dicom.ContentDate = dt.strftime(DICOM_DATE_FORMAT)
    rtstruct_dicom.ContentTime = dt.strftime(DICOM_TIME_FORMAT)
    rtstruct_dicom.InstanceCreationDate = dt.strftime(DICOM_DATE_FORMAT)
    rtstruct_dicom.InstanceCreationTime = dt.strftime(DICOM_TIME_FORMAT)
    rtstruct_dicom.SeriesDate = dt.strftime(DICOM_DATE_FORMAT)
    rtstruct_dicom.SeriesTime = dt.strftime(DICOM_TIME_FORMAT)
    rtstruct_dicom.StructureSetDate = dt.strftime(DICOM_DATE_FORMAT)
    rtstruct_dicom.StructureSetTime = dt.strftime(DICOM_TIME_FORMAT)

    return rtstruct_dicom

def add_landmarks_data(
    rtstruct_dicom: RtStructDicom,
    landmarks_data: LandmarksFrame,
    ref_cts: List[CtDicom],
    landmark_prefix: str = '') -> RtStructDicom:
    rtstruct_dicom = rtstruct_dicom.copy()
    landmarks = landmarks_data['landmark-id'].to_list()
    landmarks_data = landmarks_data[list(range(3))].to_numpy()

    # May not be any structures yet.
    if hasattr(rtstruct_dicom, 'StructureSetROISequence'):
        num_origin = len(rtstruct_dicom.StructureSetROISequence)
    else:
        rtstruct_dicom.StructureSetROISequence = []
        num_origin = 0

    for i, (id, d) in enumerate(zip(landmarks, landmarks_data)):
        # Create structure set roi.
        ss_roi = dcm.dataset.Dataset()
        ss_roi.ReferencedFrameOfReferenceUID = ref_cts[0].FrameOfReferenceUID
        ss_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        ss_roi.ROIName = f'{landmark_prefix}{id}'
        ss_roi.ROINumber = i + num_origin
        rtstruct_dicom.StructureSetROISequence.append(ss_roi)

        # Choose closest CT as the ref CT.
        zs = np.array([float(c.ImagePositionPatient[2]) for c in ref_cts])
        ref_ct = ref_cts[np.abs(zs - d[2]).argmin()]
        ref_image = dcm.dataset.Dataset()
        ref_image.ReferencedSOPClassUID = ref_ct.file_meta.MediaStorageSOPClassUID
        ref_image.ReferencedSOPInstanceUID = ref_ct.file_meta.MediaStorageSOPInstanceUID

        # Add the contour data.
        contour = dcm.dataset.Dataset()
        contour.NumberOfContourPoints = 1
        contour.ContourGeometricType = 'POINT'
        contour.ContourData = list(d)
        contour.ContourImageSequence = [ref_image]
        roi_contour = dcm.dataset.Dataset()
        roi_contour.ReferencedROINumber = i + num_origin
        roi_contour.ROIDisplayColor = [255, 255, 0]     # Yellow.
        roi_contour.ContourSequence = [contour]
        rtstruct_dicom.ROIContourSequence.append(roi_contour)

    return rtstruct_dicom

def add_regions_data(
    rtstruct_dicom: RtStructDicom,
    regions_data: RegionArrays,
    ref_cts: List[CtDicom]) -> RtStructDicom:
    # Add regions data.
    rtstruct_dicom = rtstruct_dicom.copy()
    rtstruct_dicom.RTROIObservationsSequence = []
    rtstruct_dicom.StructureSetROISequence = []
    palette = sns.color_palette('colorblind', len(regions_data.keys()))
    for i, (r, d) in enumerate(regions_data.items()):
        # Create contour.
        roi_contour = dcm.dataset.Dataset()
        roi_contour.ReferencedROINumber = str(i)
        roi_contour.ROIDisplayColor = list(np.array(palette[i]) * 255)  # Convert to 8-bit colour.

        # Add slice contours.
        roi_contour.ContourSequence = []
        for j, c in enumerate(ref_cts):
            # Get slice.
            slice_data = d[:, :, j].astype(np.uint8)
            if slice_data.sum() == 0:
                continue

            # Get contour coordinates.
            coords = get_contour_coords(slice_data)

            # Velocity has an issue with loading contours that contain less than 3 points.
            for k, cc in enumerate(coords):
                if len(cc) < 3:
                    raise ValueError(f"Contour {k} of slice {j} contains only 3 points: {cc}. Velocity will not like this.")

            # 'coords' is a list of numpy arrays, each array containing the (n x 3) coordinates of a contour.
            # I.e. multiple contours per slice are possible.
            for cc in coords:
                # Convert to patient coords.
                xy_origin = c.ImagePositionPatient[:-1]
                xy_spacing = c.PixelSpacing
                cc = xy_spacing * cc + xy_origin

                # Format contours.
                z_origin = c.ImagePositionPatient[2]
                z_idxs = np.ones((len(cc), 1)) * z_origin
                cc = np.concatenate((cc, z_idxs), axis=1)
                cc = list(cc.flatten())

                # Create contour and contour image.
                contour_image = dcm.dataset.Dataset()
                contour_image.ReferencedSOPClassUID = c.file_meta.MediaStorageSOPClassUID
                contour_image.ReferencedSOPInstanceUID = c.file_meta.MediaStorageSOPInstanceUID
                contour = dcm.dataset.Dataset()
                contour.ContourData = cc
                contour.ContourGeometricType = 'CLOSED_PLANAR'
                contour.NumberOfContourPoints = len(cc) / 3
                contour.ContourImageSequence = [contour_image]
                roi_contour.ContourSequence.append(contour)

        # Add 'roi_contour' to RTSTRUCT dicom.
        rtstruct_dicom.ROIContourSequence.append(roi_contour)

        # Create structure set roi.
        structure_set_roi = dcm.dataset.Dataset()
        structure_set_roi.ReferencedFrameOfReferenceUID = ref_cts[0].FrameOfReferenceUID
        structure_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        structure_set_roi.ROIName = r
        structure_set_roi.ROINumber = str(i)
        rtstruct_dicom.StructureSetROISequence.append(structure_set_roi)

        # Create RT roi observation.
        rt_roi_obs = dcm.dataset.Dataset()
        rt_roi_obs.ObservationNumber = str(i)
        rt_roi_obs.ReferencedROINumber = str(i)
        rt_roi_obs.ROIInterpreter = ''
        rt_roi_obs.RTROIInterpretedType = ''
        rtstruct_dicom.RTROIObservationsSequence.append(rt_roi_obs)

    return rtstruct_dicom

def get_contour_coords(
    data: CtSlice,
    method: Literal['opencv', 'skimage'] = 'skimage') -> List[np.ndarray]:  # Arrays will have different numbers of points - can't return as single numpy array.
    if method == 'opencv':
        # coords, _ = cv.findContours(slice_data, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # 'CHAIN_APPROX_SIMPLE' tries to replace straight line boundaries with two end points, instead
        # of using many points along the line - however it was producing only two points for some small
        # structures which Velocity doesn't like.
        coords, _ = cv.findContours(data, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Process results.
        # Each slice can have multiple contours - separate foreground regions in the image.
        for k, c in enumerate(coords):
            # OpenCV adds intermediate dimension - for some reason?
            c = c.squeeze(1)

            # Change in v4.11?
            # # OpenCV returns (y, x) points, so flip.
            c = np.flip(c, axis=1)
            coords[k] = c

    elif method == 'skimage':
        coords = ski.measure.find_contours(data)
        # Skimage needs no post-processing, as it returns (x, y) along the same
        # axes as the passed 'data'. Also no strange intermediate dimensions.
    else:
        raise ValueError(f"Method '{method}' not recognized. Use 'opencv' or 'skimage'.")
            
    return coords
