from mymi.typing import *

def landmarks_to_data(landarks_data: Union[LandmarksData, LandmarksVoxelData]) -> Points3D:
    return landmarks_data[list(range(3))].to_numpy()

def landmarks_to_image_coords(
    landmarks_data: LandmarksData,
    spacing: Spacing3D,
    offset: Point3D) -> LandmarksVoxelData:
    landmarks_data = landmarks_data.copy()
    lm_data = landmarks_data[list(range(3))].to_numpy()
    lm_data = np.round((lm_data - offset) / spacing).astype(int)
    landmarks_data[list(range(3))] = lm_data
    return landmarks_data

def landmarks_to_patient_coords(
    landmarks_data: LandmarksVoxelData,
    spacing: Spacing3D,
    offset: Point3D) -> LandmarksData:
    landmarks_data = landmarks_data.copy()
    lm_data = landmarks_data[list(range(3))].to_numpy()
    lm_data = lm_data * spacing + offset
    landmarks_data[list(range(3))] = lm_data
    return landmarks_data

def point_to_image_coords(
    point: Point3D,
    spacing: Spacing3D,
    offset: Point3D) -> Voxel:
    point = np.round((np.array(point) - offset) / spacing).astype(int)
    return tuple(point)
