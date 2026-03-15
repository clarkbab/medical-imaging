from mymi.typing import *

from .affine import affine_origin, affine_spacing

def landmarks_from_data(data: Points3D) -> LandmarksFrame:
    return pd.DataFrame(data).rename_axis('landmark-id').reset_index()

def landmarks_to_data(data: Union[LandmarksFrame, LandmarksFrameVox]) -> Points3D:
    return data[list(range(3))].to_numpy()

def landmarks_to_image_coords(
    data: LandmarksFrame,
    affine: Affine,
    ) -> LandmarksFrameVox:
    data = data.copy()
    lm_data = data[list(range(3))].to_numpy()
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    lm_data = np.round((lm_data - origin) / spacing).astype(int)
    data[list(range(3))] = lm_data
    return data

def landmarks_to_world_coords(
    data: LandmarksFrameVox,
    affine: Affine,
    ) -> LandmarksFrame:
    data = data.copy()
    lm_data = data[list(range(3))].to_numpy()
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    lm_data = lm_data * spacing + origin
    data[list(range(3))] = lm_data
    return data

def point_to_image_coords(
    point: Point3D,
    affine: Affine,
    ) -> Voxel:
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    voxel = tuple(float(p) for p in np.round((np.array(point) - origin) / spacing).astype(int))
    return voxel

def voxel_to_world_coords(
    voxel: Voxel,
    affine: Affine,
    ) -> Point3D:
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    point = tuple(float(v) * s + o for v, s, o in zip(voxel, spacing, origin))
    return point

def replace_landmarks(
    data: LandmarksFrame,
    points: Points3D) -> LandmarksFrame:
    assert len(data) == len(points), "Number of points must match number of landmarks."
    data = data.copy()
    data[list(range(3))] = points
    return data
