from mymi.typing import *

def landmarks_from_data(data: Points3D) -> LandmarksFrame:
    return pd.DataFrame(data).rename_axis('landmark-id').reset_index()

def landmarks_to_data(data: Union[LandmarksFrame, LandmarksFrameVox]) -> Points3D:
    return data[list(range(3))].to_numpy()

def landmarks_to_image_coords(
    data: LandmarksFrame,
    spacing: Spacing3D,
    origin: Point3D) -> LandmarksFrameVox:
    data = data.copy()
    lm_data = data[list(range(3))].to_numpy()
    lm_data = np.round((lm_data - origin) / spacing).astype(int)
    data[list(range(3))] = lm_data
    return data

def landmarks_to_patient_coords(
    data: LandmarksFrameVox,
    spacing: Spacing3D,
    origin: Point3D) -> LandmarksFrame:
    data = data.copy()
    lm_data = data[list(range(3))].to_numpy()
    lm_data = lm_data * spacing + origin
    data[list(range(3))] = lm_data
    return data

def point_to_image_coords(
    point: Point3D,
    spacing: Spacing3D,
    origin: Point3D) -> Voxel:
    point = np.round((np.array(point) - origin) / spacing).astype(int)
    return tuple(point)

def replace_landmarks(
    data: LandmarksFrame,
    points: Points3D) -> LandmarksFrame:
    assert len(data) == len(points), "Number of points must match number of landmarks."
    data = data.copy()
    data[list(range(3))] = points
    return data
