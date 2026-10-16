from dicomset.typing import *
from dicomset.utils import affine_spacing, affine_origin, create_affine, fov_width, \
    to_image_coords, affine_subset, plot_slice, resample
import numpy as np

def rotate_point(x, theta):
    theta = np.radians(theta)
    Rz = np.array([[np.cos(theta),  np.sin(theta), 0],
                   [-np.sin(theta), np.cos(theta), 0],
                   [0,              0,             1]])
    return Rz @ x

# Given the reconstructed volume, extracts the GTV slice through
# the detector geometry.
def extract_gtv_view(
    reconst: Image3D,
    affine: AffineMatrix3D,
    gtv_centroid: Point3D,  # The location of the GTV in the reconst volume.
    sdd: float,
    sid: float,
    det_size: Size2D,
    det_affine: AffineMatrix2D,
    fill: Number | Literal['max', 'mean', 'median', 'min'] = 'min',
    ) -> Image2D:
    
    # Convert detector geometry to GTV plane.
    gid = sid + gtv_centroid[1]  # Distance from source to GTV plane - positive y in reconstruction is towards detector.
    det_spacing = affine_spacing(det_affine)
    det_origin = affine_origin(det_affine)
    ds_g = det_spacing * (gid / sdd)
    do_g = det_origin * (gid / sdd)
    da_g = create_affine(ds_g, (do_g[0], do_g[1]))
    dfw_g = fov_width(det_size, a=da_g)
    # print(dfw_g)
    dm_g = do_g + np.array(dfw_g)
    
    # Extract single slice.
    gtv_im = to_image_coords(gtv_centroid, affine)
    s = reconst[:, gtv_im[1]]
    s_affine = affine_subset(affine, dims=[0, 2])
    points = np.array([
        [do_g[0], do_g[1]],
        [dm_g[0], dm_g[1]],
    ])
    # print(points)
    # plot_slice(s, a=s_affine, p=points, pa=1.0, pc='red')
    
    # Resample slice to match detector size.
    sc = resample(s, affine=s_affine, output_affine=da_g, output_size=det_size, fill=fill)

    return sc, da_g
