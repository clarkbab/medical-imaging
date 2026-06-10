import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, gaussian_filter, gaussian_filter1d

from mymi.transforms import resample

def interpolate_z(data: np.ndarray) -> np.ndarray:
    # Find "missing" slices.
    extent = extent(data)
    z_min = extent[0][2]
    z_max = extent[1][2]

    # Interpolate using ground truth data only, i.e. don't update the 
    # data as we go.
    new_data = data.copy()
    for z in range(z_min, z_max + 1):
        data_z = data[:, :, z]
        if data_z.sum() != 0:
            continue
            
        # Find closest non-empty slices.
        max_diff = 5
        data_below = None
        data_above = None
        for i in range(max_diff):
            if data_below is None and data[:, :, (z - i - 1)].sum() != 0:
                data_below = data[:, :, (z - i - 1)]
            if data_above is None and data[:, :, (z + i + 1)].sum() != 0:
                data_above = data[:, :, (z + i + 1)]
            if data_below is not None and data_above is not None:
                break
                
        if data_below is None or data_above is None:
            raise ValueError("")
            
        # Interpolate from surrounding slices.
        data_z_around = np.stack((data_below, data_above), axis=2)
        spacing = (1, 1, 1)
        output_spacing = (1, 1, 0.5)
        output_size = (*data.shape[0:2], 3)
        data_z_resampled = resample(data_z_around, output_size=output_size, output_spacing=output_spacing, spacing=spacing)
        
        # Replace slice.
        new_data[:, :, z] = data_z_resampled[:, :, 1]

    return new_data


def interpolate_masks_radial(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    t: float,
    n_angles: int = 720,
    return_radial_functions: bool = False,
    smooth: bool = True,
) -> np.ndarray:
    """Radial interpolation. Exact for convex shapes (rectangles, ellipses, etc.).
    Requires both shapes to be star-shaped w.r.t. their centroids."""
    shape = mask_a.shape

    ca = np.argwhere(mask_a).mean(axis=0)
    cb = np.argwhere(mask_b).mean(axis=0)
    ct = (1 - t) * ca + t * cb

    def radial_boundary(mask, center):
        pts = np.argwhere(mask).astype(float)
        dr = pts[:, 0] - center[0]
        dc = pts[:, 1] - center[1]
        theta = np.arctan2(dr, dc) % (2 * np.pi)
        r = np.hypot(dr, dc)
        bin_idx = (theta / (2 * np.pi) * n_angles).astype(int) % n_angles
        radii = np.full(n_angles, np.nan)
        for idx, val in zip(bin_idx, r):
            if np.isnan(radii[idx]) or val > radii[idx]:
                radii[idx] = val
        angles = np.arange(n_angles)
        nan_mask = np.isnan(radii)
        if np.any(nan_mask):
            valid = ~nan_mask
            radii[nan_mask] = np.interp(angles[nan_mask], angles[valid], radii[valid], period=n_angles)
        if smooth:
            radii = gaussian_filter1d(radii, sigma=2, mode='wrap')
        return radii

    ra = radial_boundary(mask_a, ca)
    rb = radial_boundary(mask_b, cb)
    rt = (1 - t) * ra + t * rb

    rows, cols = np.mgrid[:shape[0], :shape[1]]
    dr = rows - ct[0]
    dc = cols - ct[1]
    theta = np.arctan2(dr, dc) % (2 * np.pi)
    pixel_r = np.hypot(dr, dc)
    bin_idx = (theta / (2 * np.pi) * n_angles).astype(int) % n_angles
    mask = pixel_r <= rt[bin_idx]
    if return_radial_functions:
        return mask, ra, rb, rt
    return mask


def interpolate_masks_radial_3d(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    t: float,
    n_azimuth: int = 180,
    n_elevation: int = 90,
    return_radial_functions: bool = False,
    smooth: bool = True,
) -> np.ndarray:
    """Radial interpolation for 3D masks using spherical coordinates.
    Requires both shapes to be star-shaped w.r.t. their centroids."""
    shape = mask_a.shape

    ca = np.argwhere(mask_a).mean(axis=0)
    cb = np.argwhere(mask_b).mean(axis=0)
    ct = (1 - t) * ca + t * cb

    def radial_boundary_3d(mask, center):
        pts = np.argwhere(mask).astype(float)
        dx = pts[:, 0] - center[0]
        dy = pts[:, 1] - center[1]
        dz = pts[:, 2] - center[2]
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        azimuth = np.arctan2(dy, dx) % (2 * np.pi)
        elevation = np.arccos(np.clip(dz / (r + 1e-8), -1, 1))
        az_idx = (azimuth / (2 * np.pi) * n_azimuth).astype(int) % n_azimuth
        el_idx = (elevation / np.pi * n_elevation).astype(int) % n_elevation
        radii = np.full((n_azimuth, n_elevation), np.nan)
        for a, e, val in zip(az_idx, el_idx, r):
            if np.isnan(radii[a, e]) or val > radii[a, e]:
                radii[a, e] = val
        grid_az, grid_el = np.meshgrid(np.arange(n_azimuth), np.arange(n_elevation), indexing='ij')
        nan_mask = np.isnan(radii)
        if np.any(nan_mask):
            valid = ~nan_mask
            points = np.stack([grid_az[valid], grid_el[valid]], axis=-1)
            values = radii[valid]
            points_full = np.stack([grid_az[nan_mask], grid_el[nan_mask]], axis=-1)
            radii[nan_mask] = griddata(points, values, points_full, method='nearest')
        if smooth:
            radii = gaussian_filter(radii, sigma=2, mode='wrap')
        return radii

    ra = radial_boundary_3d(mask_a, ca)
    rb = radial_boundary_3d(mask_b, cb)
    rt = (1 - t) * ra + t * rb

    x, y, z = np.mgrid[:shape[0], :shape[1], :shape[2]]
    dx = x - ct[0]
    dy = y - ct[1]
    dz = z - ct[2]
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    azimuth = np.arctan2(dy, dx) % (2 * np.pi)
    elevation = np.arccos(np.clip(dz / (r + 1e-8), -1, 1))
    az_idx = (azimuth / (2 * np.pi) * n_azimuth).astype(int) % n_azimuth
    el_idx = (elevation / np.pi * n_elevation).astype(int) % n_elevation
    mask = r <= rt[az_idx, el_idx]
    if return_radial_functions:
        return mask, ra, rb, rt
    return mask
