from mymi.typing import *

def landmarks_to_data(data: Union[LandmarksData, LandmarksVoxelData]) -> Points3D:
    return data[list(range(3))].to_numpy()

def landmarks_to_image_coords(
    data: LandmarksData,
    spacing: Spacing3D,
    offset: Point3D) -> LandmarksVoxelData:
    data = data.copy()
    lm_data = data[list(range(3))].to_numpy()
    lm_data = np.round((lm_data - offset) / spacing).astype(int)
    data[list(range(3))] = lm_data
    return data

def landmarks_to_patient_coords(
    data: LandmarksVoxelData,
    spacing: Spacing3D,
    offset: Point3D) -> LandmarksData:
    data = data.copy()
    lm_data = data[list(range(3))].to_numpy()
    lm_data = lm_data * spacing + offset
    data[list(range(3))] = lm_data
    return data

def point_to_image_coords(
    point: Point3D,
    spacing: Spacing3D,
    offset: Point3D) -> Voxel:
    point = np.round((np.array(point) - offset) / spacing).astype(int)
    return tuple(point)

def replace_landmarks(
    data: LandmarksData,
    points: Points3D) -> LandmarksData:
    assert len(data) == len(points), "Number of points must match number of landmarks."
    data = data.copy()
    data[list(range(3))] = points
    return data
