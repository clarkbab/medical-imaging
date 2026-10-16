from augmed.typing import *
from augmed.utils import to_tensor, to_numpy, to_tuple
from dicomset.utils import logger, hist_eq as hist_eq_fn, plot_slice, minmax
from dicomset.utils.args import arg_to_list
from dicomset.utils.geometry import affine_origin, affine_spacing, create_affine, fov_centre
from dicomset.typing import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydicom as dcm
import seaborn as sns
from scipy.ndimage import binary_dilation
from skimage.restoration import inpaint_biharmonic
import subprocess
import tempfile
import torch
import torchio as tio
from tqdm import tqdm
from typing import *

from mymi.utils.ct import fill_ct_background, has_ct_background, hu_to_attenuation
from mymi.typing import *
from mymi.utils.io import sitk_load_volume, sitk_save_volume

def project_diffdrr(
    volume: Image3D,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    couch_shift: Point3D = (0.0, 0.0, 0.0),
    fill_background: bool = True,
    labels: BatchLabelImage3D | None = None,
    labels_output: Literal['boolean', 'density'] = 'boolean',
    n_angles_batch: int | None = 20,
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = False,
    *,
    patch_size: int = 128,
    ) -> BatchImage2D | Tuple[BatchImage2D, BatchChannelLabelImage2D]:
    from diffdrr.data import read
    from diffdrr.drr import DRR

    if verbose:
        logger.log_method('Creating DiffDRR projections')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    volume = to_tensor(volume, device=device, dtype=torch.float32)
    affine = to_tensor(affine, device=volume.device, dtype=torch.float32)
    treatment_iso = to_tensor(treatment_iso, device=volume.device, dtype=torch.float32)
    det_size = to_tensor(det_size, device=volume.device, dtype=torch.float32)
    det_spacing = to_tensor(det_spacing, device=volume.device, dtype=torch.float32)
    det_offset = to_tensor(det_offset, device=volume.device, dtype=torch.float32)
    couch_shift = to_tensor(couch_shift, device=volume.device, dtype=torch.float32)
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
        labels = to_tensor(labels, device=volume.device, dtype=torch.bool)
    kv_source_angles = arg_to_list(kv_source_angles, float)
    if not kv_source_angles:
        raise ValueError('kv_source_angles is empty — cannot project with zero angles')
    # DiffDRR uses the detector angles.
    # 'machine doesn't matter for this conversion'
    kv_detector_angles = convert_angles(kv_source_angles, from_='kv-source', to='kv-detector', machine='varian')

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if fill_background and has_ct_background(volume):
        volume = fill_ct_background(volume, affine)

    # Convert from Hounsfield Units to (relative) attenuation coefficients, as for the
    # other projection methods. This replaces DiffDRR's own HU -> density mapping, which
    # normalises intensities to [0, 1] and so gives projections on a different scale.
    attenuation = hu_to_attenuation(volume)

    # Move the treatment isocentre to the origin.
    spacing = to_numpy(affine_spacing(affine))
    origin = to_numpy(affine_origin(affine))
    origin = origin - to_numpy(treatment_iso) - to_numpy(couch_shift)
    tio_affine = create_affine(spacing, origin)

    # DiffDRR still needs a (HU) volume to read the geometry from - the density is replaced below.
    tio_volume = tio.ScalarImage(tensor=volume.cpu().unsqueeze(0), affine=tio_affine)
    subject = read(tio_volume, center_volume=False)

    drr = DRR(
        subject,
        sdd=sdd,
        width=int(det_size[0]),
        height=int(det_size[1]),
        patch_size=patch_size,
        delx=float(det_spacing[0]),
        dely=float(det_spacing[1]),
        x0=float(det_offset[0]),
        y0=float(det_offset[1]),
        reverse_x_axis=False,
    ).to(volume.device)

    # Channel 0 is the CT, the remaining channels are the labels. Each channel is rendered by
    # swapping the DRR's density volume (i.e. rather than using DiffDRR's mask channels).
    n_labels = len(labels) if labels is not None else 0
    def channel_density(c: int) -> torch.Tensor:
        if c == 0:
            return attenuation
        label = labels[c - 1].to(torch.float32)
        if labels_output == 'density':
            # Extract the masked region from the volume (i.e. label * volume) so that
            # the projected label channel represents the density of the masked region.
            label = label * attenuation
        return label

    # Split angles into batches if requested.
    if n_angles_batch is None:
        batches = [kv_detector_angles]
    else:
        batches = [kv_detector_angles[i:i + n_angles_batch] for i in range(0, len(kv_detector_angles), n_angles_batch)]

    chunks = [[] for _ in range(1 + n_labels)]
    n_batches = len(batches)
    for i, angles in enumerate(tqdm(batches, desc="DiffDRR angle batches")):
        rot = torch.tensor([
            [a, 0.0, 0.0] for a in angles
        ], device=volume.device, dtype=torch.float32)
        trans = torch.tensor([
            [0, sid, 0.0] for _ in angles
        ], device=volume.device, dtype=torch.float32)

        for c in range(1 + n_labels):
            drr.density = channel_density(c)
            with torch.no_grad():
                proj_data = drr(rot, trans, parameterization="euler_angles", convention="ZYX", degrees=True)

            # Correct x/y axes and move to CPU immediately.
            chunks[c].append(proj_data.moveaxis(2, 3)[:, 0].cpu())
            del proj_data

        if progress_callback:
            progress_callback(i + 1, n_batches)

        # Free GPU memory before next batch.
        del rot, trans
        torch.cuda.empty_cache()

    # Concatenate batches along the angles dimension.
    projs = [torch.cat(c, dim=0) for c in chunks]
    del chunks

    # This was required to match treatment images. I believe this is because
    # the projection codes assumes a cameras-eye-view instead of beams-eye-view.
    projs = [torch.flip(p, dims=[1, -1]) for p in projs]

    volume_proj = projs[0]
    if labels is not None:
        labels_proj = torch.stack(projs[1:], dim=1)
        if labels_output == 'boolean':
            # Make a pixel foreground if its ray passed through any foreground
            # voxel (with some threshold for floating point noise).
            labels_proj = labels_proj > 1e-3 * float(spacing.min())
        return volume_proj, labels_proj
    else:
        return volume_proj

# I think this actually takes kv detector angles, see contour alignment tool behaviour.
def project_igt(
    volume: Image3D,
    affine: AffineMatrix3D,
    # This is the point (in planning CT coords) around which the projections are created.
    # That's if we're trying to simulate projections that would be acquired during treatment.
    # For CBCT, we might need to apply a treatment iso offset.
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],
    # For half-fan CBCT, the patient is centred laterally before the CBCT image is acquired to
    # avoid collision.
    couch_shift: Point3D = (0.0, 0.0, 0.0),
    fill_background: bool = True,
    labels: BatchLabelImage3D | None = None,
    labels_output: Literal['boolean', 'density'] = 'boolean',
    # If more than one batch, the geometry/projection files for each batch are saved in 'batch_*' subfolders.
    n_angles_batch: int | None = 20,
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = False,
    *,
    # We need a folder to save the intermediate (.mha, .csv, .xml) files.
    dirpath: DirPath | None = None,
    recreate: bool = True,
    ) -> BatchImage2D | Tuple[BatchImage2D, BatchChannelLabelImage2D]:
    if verbose:
        logger.log_method('Creating IGT projections')
    if dirpath is None:
        dirpath = tempfile.gettempdir()
    volume = to_numpy(volume)
    affine = to_numpy(affine)
    if labels is not None:
        labels = to_numpy(labels).astype(np.bool_)
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
        assert labels.shape[1:] == volume.shape, f"Expected labels to have same spatial shape \
            as volume, got {labels.shape[1:]} but expected {volume.shape}"
    treatment_iso = [float(v) for v in treatment_iso]
    couch_shift = [float(v) for v in couch_shift]
    det_size = [int(v) for v in det_size]
    det_spacing = [float(v) for v in det_spacing]
    det_offset = [float(v) for v in det_offset]
    kv_source_angles = arg_to_list(kv_source_angles, float)
    if not kv_source_angles:
        raise ValueError('kv_source_angles is empty — cannot project with zero angles')

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if fill_background and has_ct_background(volume):
        volume = fill_ct_background(volume, affine)

    # Convert from Hounsfield Units to (relative) attenuation coefficients, as for the
    # other projection methods.
    volume = to_numpy(hu_to_attenuation(volume))

    # Label channels are projected as separate volumes.
    if labels is not None:
        rois = labels.astype(np.float32)
        if labels_output == 'density':
            # Extract the masked region from the volume (i.e. label * volume) so that
            # the projected label channel represents the density of the masked region.
            rois = rois * volume
    else:
        rois = None

    # Does IGT use a counter-clockwise positive, (CCW+) angular convention?
    # Needed this to look right.
    kv_source_angles = reverse_angles(kv_source_angles)
    # IGT uses kv_detector angles.
    # 'machine doesn't matter for this conversion'
    kv_detector_angles = convert_angles(kv_source_angles, from_='kv-source', to='kv-detector', machine='varian')

    # Flip data along z axis - required by IGT.
    # This was something that just worked.
    # I think this is because the .xim files used a CCW+ angular convention,
    # and one way to achieve this in a CW+ convention is to flip the z-axis.
    volume = np.flip(volume, axis=2)
    if rois is not None:
        rois = np.flip(rois, axis=-1)

    # Move the volume to the treatment isocentre.
    spacing = to_numpy(affine_spacing(affine))
    min_spacing = float(np.min(spacing))
    origin = to_numpy(affine_origin(affine))
    origin = np.array(origin)
    origin[0] = origin[0] - treatment_iso[0] - couch_shift[0]
    origin[1] = origin[1] - treatment_iso[1] - couch_shift[1]
    # Need to adjust the z origin because of flipped axis.
    max_z_loc = origin[2] + (volume.shape[2] - 1) * spacing[2]
    origin[2] = treatment_iso[2] + couch_shift[2] - max_z_loc

    # Reorder axes to (x, z, y) - required by IGT.
    # Is this IEC geometry?
    volume = np.moveaxis(volume, 1, 2)
    spacing = (spacing[0], spacing[2], spacing[1])
    origin = (origin[0], origin[2], origin[1])
    affine = create_affine(spacing, origin)
    if rois is not None:
        rois = np.moveaxis(rois, -2, -1)

    # Create the CT and ROI MHA files (shared by all angle batches).
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    ct_filepath = os.path.join(dirpath, 'CT.mha')
    if not os.path.exists(ct_filepath) or recreate:
        sitk_save_volume(volume, affine, ct_filepath)
    n_labels = len(rois) if rois is not None else 0
    if rois is not None:
        for i in range(n_labels):
            roi_filepath = os.path.join(dirpath, f'ROI_{i}.mha')
            if not os.path.exists(roi_filepath) or recreate:
                sitk_save_volume(rois[i], affine, roi_filepath)

    # Split angles into batches if requested.
    if n_angles_batch is None:
        batches = [kv_detector_angles]
    else:
        batches = [kv_detector_angles[i:i + n_angles_batch] for i in range(0, len(kv_detector_angles), n_angles_batch)]
    n_batches = len(batches)

    def forward_project(
        in_filepath: FilePath,
        out_filepath: FilePath,
        geometry_filepath: FilePath,
        ) -> None:
        cmd = [
            r"D:\Brett\code\IGT\IGTExecutables\VS2013_x64\rtkforwardprojections.exe",
            "-i", in_filepath,
            "-o", out_filepath,
            "-g", geometry_filepath,
            "--fp", "Joseph",
            "--dimension", f"{det_size[0]},{det_size[1]}",
            "--spacing", str(det_spacing[0]),
        ]
        print(cmd)
        subprocess.run(cmd)

    # Channel 0 is the CT, the remaining channels are the labels.
    chunks = [[] for _ in range(1 + n_labels)]
    for b, angles in enumerate(tqdm(batches, desc="IGT angle batches")):
        batch_dirpath = dirpath if n_batches == 1 else os.path.join(dirpath, f'batch_{b:04d}')
        if not os.path.exists(batch_dirpath):
            os.makedirs(batch_dirpath)

        # If existing angles file doesn't match, then recreate.
        batch_recreate = recreate
        angles_filepath = os.path.join(batch_dirpath, 'angles.csv')
        if os.path.exists(angles_filepath):
            df = pd.read_csv(angles_filepath, header=None)
            if len(df[0].tolist()) != len(angles) or not np.allclose(df[0].tolist(), angles):
                print(f"Existing angles file {angles_filepath} doesn't match provided angles, recreating projections.")
                batch_recreate = True

        # Write angles to CSV.
        if not os.path.exists(angles_filepath) or batch_recreate:
            with open(angles_filepath, 'w') as f:
                for a in angles:
                    f.write(f"{a}\n")

        # Create geometry file.
        geometry_filepath = os.path.join(batch_dirpath, 'geometry.xml')
        if not os.path.exists(geometry_filepath) or batch_recreate:
            cmd = [
                r"D:\Brett\code\IGT\IGTExecutables\VS2013_x64\igtsimulatedgeometry.exe",
                "-i", angles_filepath,
                "--sid", str(sid),
                "--sdd", str(sdd),
                "--proj_iso_x", str(det_offset[0]),
                "--proj_iso_y", str(det_offset[1]),
                "-o", geometry_filepath,
            ]
            print(cmd)
            subprocess.run(cmd)

        # Forward project the CT and ROIs.
        for c in range(1 + n_labels):
            name = 'CT' if c == 0 else f'ROI_{c - 1}'
            fp_filepath = os.path.join(batch_dirpath, f'{name}_FP.mha')
            if not os.path.exists(fp_filepath) or batch_recreate:
                forward_project(os.path.join(dirpath, f'{name}.mha'), fp_filepath, geometry_filepath)

            # Load results.
            proj, _ = sitk_load_volume(fp_filepath)
            chunks[c].append(torch.from_numpy(np.moveaxis(proj, -1, 0).astype(np.float32)))

        if progress_callback:
            progress_callback(b + 1, n_batches)

    # Concatenate batches along the angles dimension.
    projs = [torch.cat(c, dim=0) for c in chunks]
    del chunks

    # This was required to match treatment images. I believe this is because
    # the projection codes assumes a cameras-eye-view instead of beams-eye-view.
    projs = [torch.flip(p, dims=[1, -1]) for p in projs]

    volume_proj = projs[0]
    if labels is not None:
        labels_proj = torch.stack(projs[1:], dim=1)
        if labels_output == 'boolean':
            # Make a pixel foreground if its ray passed through any foreground
            # voxel (with some threshold for floating point noise).
            labels_proj = labels_proj > 1e-3 * min_spacing
        return volume_proj, labels_proj
    else:
        return volume_proj

def project_points(
    points: Points3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],
    ) -> BatchPixels:
    """Project 3D points to 2D pixel coordinates for a rotating kV source / flat-panel detector.

    Geometry (IEC61217):
      - kv_source_angle θ=0°: source directly above patient (+y), rotates clockwise (CW+) viewed from +z.
      - Source position at θ: S = (sid·sin θ, sid·cos θ, 0) relative to isocentre.
      - Detector u-axis (horizontal): (cos θ, -sin θ, 0).
      - Detector v-axis (vertical):   (0, 0, 1)  — positive toward patient head.
      - det_offset: physical shift of the detector panel from the beam axis (mm).
        Positive u0 shifts the panel in the +u direction (half-fan).

    Returns
    -------
    pixels : (n_angles, N, 2) pixel coordinates (u, v), 0-indexed, (0,0) at top-left.
        Points behind the source plane are returned as NaN.
    """
    points, return_type = to_tensor(points, dtype=torch.float64, return_type=True)
    if points.ndim == 1:
        points = points.unsqueeze(0)                    # (1, 3)
    treatment_iso = to_tensor(treatment_iso, dtype=torch.float64)
    kv_source_angles = arg_to_list(kv_source_angles, float)

    n_angles = len(kv_source_angles)
    nu, nv = int(det_size[0]), int(det_size[1])
    du, dv = float(det_spacing[0]), float(det_spacing[1])
    u0, v0 = float(det_offset[0]), float(det_offset[1])

    # Translate points into the isocentre frame.
    pts = points - treatment_iso                        # (n_points, 3)

    thetas = torch.tensor(np.deg2rad(kv_source_angles), dtype=torch.float64)
    cos_t  = torch.cos(thetas)                          # (n_angles,)
    sin_t  = torch.sin(thetas)
    zeros  = torch.zeros(n_angles, dtype=torch.float64)

    # Source positions: (n_angles, 3)
    sources = torch.stack([sid * sin_t, sid * cos_t, zeros], dim=1)
    # Detector u-axis: (n_angles, 3)
    u_hats  = torch.stack([cos_t, -sin_t, zeros], dim=1)
    # Beam normal (source → isocentre): (n_angles, 3)
    beam_ns = -sources / sid

    v_hat   = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    pixels  = torch.full((n_angles, points.shape[0], 2), float('nan'), dtype=torch.float64)

    for i in range(n_angles):
        src      = sources[i]   # (3,)
        u_hat_i  = u_hats[i]   # (3,)
        beam_n_i = beam_ns[i]  # (3,)

        # Vectors from source to each point: (n_points, 3)
        d = pts - src

        # Projection distance along beam axis — must be > 0 for a valid projection.
        dot_beam = d @ beam_n_i                         # (n_points,)
        mag = sdd / dot_beam
        mag[dot_beam <= 0] = float('nan')

        # Position on detector plane in mm (origin = beam axis).
        u_mm = mag * (d @ u_hat_i)
        v_mm = mag * (d @ v_hat)

        # Convert to pixel coordinates; det_offset shifts the panel from the beam axis.
        pixels[i, :, 0] = (u_mm - u0) / du + nu / 2
        pixels[i, :, 1] = (v_mm - v0) / dv + nv / 2

    if return_type is np.ndarray:
        pixels = to_numpy(pixels)
    return pixels

def project_ctorch(
    volume: Image3D,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    couch_shift: Point3D = (0.0, 0.0, 0.0),
    fill_background: bool = True,
    labels: BatchLabelImage3D | None = None,
    labels_output: Literal['boolean', 'density'] = 'boolean',
    n_angles_batch: int | None = 20,
    projection_method: Literal['DD', 'SF', 'ray', 'voxel'] = 'ray',
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = False,
    ) -> BatchImage2D | Tuple[BatchImage2D, BatchChannelLabelImage2D]:
    from CTorch.utils.geometry import CircGeom3D
    from CTorch.projector.projector_interface import Projector

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    volume = to_tensor(volume, device=device, dtype=torch.float32)
    affine = to_tensor(affine, device=volume.device, dtype=torch.float32)
    treatment_iso = to_tensor(treatment_iso, device=volume.device, dtype=torch.float32)
    det_size = to_tensor(det_size, device=volume.device, dtype=torch.float32)
    det_spacing = to_tensor(det_spacing, device=volume.device, dtype=torch.float32)
    det_offset = to_tensor(det_offset, device=volume.device, dtype=torch.float32)
    couch_shift = to_tensor(couch_shift, device=volume.device, dtype=torch.float32)
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
        labels = to_tensor(labels, device=volume.device, dtype=torch.bool)
    kv_source_angles = arg_to_list(kv_source_angles, float)
    kv_source_angles = to_numpy(kv_source_angles)
    # Ctorch uses kV detector angles.
    # 'machine doesn't matter for this conversion'
    kv_detector_angles = convert_angles(kv_source_angles, from_='kv-source', to='kv-detector', machine='varian')
    if verbose:
        logger.log_method('Creating CTorch projections')
    # TODO: Check which voxel is the treatment isocentre, if it's too
    # far from centre the DRR might look whacky.
    # treatment_iso_vox = point_to_image_coords(treatment_iso, affine)
    # print(f"Treatment isocentre (image coords): {treatment_iso_vox}, should be near\
    #     image centre voxel {tuple(s//2 for s in volume.shape)}.")

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if fill_background and has_ct_background(volume):
        volume = fill_ct_background(volume, affine)

    # Convert from Hounsfield Units to (relative) attenuation coefficients so that the
    # projected line integral is non-negative and physically sensible, rather than a line
    # integral of HU which is dominated by the air/water HU offset (see 'hu_to_attenuation').
    volume = hu_to_attenuation(volume)

    nx, ny, nz = volume.shape
    dx, dy, dz = [float(v) for v in affine_spacing(affine)]
    nu, nv = int(det_size[0]), int(det_size[1])
    du, dv = float(det_spacing[0]), float(det_spacing[1])
    detType = 'flat' #'curve'
    # Convert angles to rad and use CCW+ system.
    kv_detector_angles = [np.deg2rad(-a) for a in kv_detector_angles]
    SAD, SDD = [sid], [sdd] # source-axis-distance, source-detector-distance

    image_centre = fov_centre(volume.shape, affine=affine)
    offset = np.array([float(v) for v in image_centre]) - np.array([float(v) for v in treatment_iso]) - np.array([float(v) for v in couch_shift])
    offset[2] = -offset[2]  # Z-axis is flipped.
    xOfst, yOfst, zOfst = [float(offset[0])], [float(offset[1])], [float(offset[2])] # image center offset
    uOfst, vOfst = [float(det_offset[0])], [float(det_offset[1])] # detecor center offset
    xSrc, zSrc = [0.0], [0.0]

    # Prepare volume tensor (shared across batches).
    volume_tensor = to_tensor(volume, dtype=torch.float32)
    if labels is not None:
        labels_tensor = to_tensor(labels, dtype=torch.float32)
        if labels_output == 'density':
            # Extract the masked region from the volume (i.e. label * volume) so that
            # the projected label channel represents the density of the masked region.
            labels_tensor = labels_tensor * volume_tensor
        volume_tensor = torch.concat([volume_tensor.unsqueeze(0), labels_tensor], axis=0).unsqueeze(0).cuda()
    else:
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).cuda()
    # Flip z, transpose x/y/z.
    volume_tensor = volume_tensor.flip(4).permute((0, 1, 4, 3, 2))

    # Split angles into batches if requested.
    if not kv_detector_angles:
        raise ValueError('kv_source_angles is empty — cannot project with zero angles')
    if n_angles_batch is None:
        batches = [kv_detector_angles]
    else:
        batches = [kv_detector_angles[i:i + n_angles_batch] for i in range(0, len(kv_detector_angles), n_angles_batch)]

    volume_proj_chunks = []
    labels_proj_chunks = [] if labels is not None else None

    n_batches = len(batches)
    for i, angles in enumerate(tqdm(batches, desc="CTorch angle batches")):
        geom = CircGeom3D(
             nx, ny, nz, dx, dy, dz, nu, nv, len(angles), angles, du, dv, detType, SAD, SDD,
             xOfst=xOfst, yOfst=yOfst, zOfst=zOfst, uOfst=uOfst, vOfst=vOfst,
             xSrc=xSrc, zSrc=zSrc, fixed=True)

        projector = Projector(geom, 'proj', projection_method, 'forward')
        with torch.no_grad():
            proj_data = projector(volume_tensor)

        vp = proj_data[0, 0].permute((0, 2, 1)).cpu()
        volume_proj_chunks.append(vp)

        if labels is not None:
            lp = proj_data[0, 1:].permute((1, 0, 3, 2)).cpu()
            if labels_output == 'boolean':
                # Make a pixel foreground if its ray passed through any foreground
                # voxel (with some threshold for floating point noise).
                lp = lp > 1e-3 * min(dx, dy, dz)
            labels_proj_chunks.append(lp)

        if progress_callback:
            progress_callback(i + 1, n_batches)

        # Free GPU memory before next batch.
        del proj_data, projector, geom
        torch.cuda.empty_cache()

    # Concatenate batches along the angles dimension.
    volume_proj = torch.cat(volume_proj_chunks, dim=0)
    del volume_proj_chunks
    # This was required to match treatment images. I believe this is because
    # the projection codes assumes a cameras-eye-view instead of beams-eye-view.
    volume_proj = torch.flip(volume_proj, dims=[1])
    volume_proj = torch.flip(volume_proj, dims=[-1])

    if labels is not None:
        labels_proj = torch.cat(labels_proj_chunks, dim=0)
        labels_proj = torch.flip(labels_proj, dims=[2])
        labels_proj = torch.flip(labels_proj, dims=[-1])
        del labels_proj_chunks
        return volume_proj, labels_proj
    else:
        return volume_proj

def load_dicom_projections(
    filepath: FilePath,
    ) -> Tuple[SliceBatch, Dict[str, Any]]:
    ds = dcm.dcmread(filepath)
    assert ds.PatientPosition == 'HFS'
    assert ds.RTImageOrientation == [1, 0, 0, 0, -1, 0]

    # Add info.
    info = {}
    info['sid'] = float(ds.RadiationMachineSAD)     
    info['sdd'] = float(ds.RTImageSID)
    info['det-spacing'] = tuple(float(f) for f in ds.ImagePlanePixelSpacing)
    info['det-offset'] = tuple(float(o) for o in ds.XRayImageReceptorTranslation[:2])

    kv_detector_angles = []
    for i, f in enumerate(ds.ExposureSequence):
        frame = int(f.ReferencedFrameNumber)
        assert frame == i + 1
        angle = float(getattr(f, "GantryAngle", np.nan))
        kv_detector_angles.append(angle)
    info['kv-detector-angles'] = kv_detector_angles

    # Load pixel data.
    data = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    data = data * slope + intercept

    # Transpose x/y axes.
    data = np.moveaxis(data, 1, 2)
    info['det-size'] = data.shape[1:]

    return data, info

# Converts between different angle types for different machines.
# This assumes that angles are already in IEC61217 format.
def convert_angles(
    angles: float | List[float],
    from_: Literal['kv-detector', 'kv-source', 'mv-detector', 'mv-source'],
    to: Literal['kv-detector', 'kv-source', 'mv-detector', 'mv-source'],
    machine: Literal['elekta', 'varian'],
    scale: Literal['degrees', 'radians'] = 'degrees',
    ) -> List[float]:
    angles, was_single = arg_to_list(angles, (int, float), return_matched=True)
    angle_360 = 2 * np.pi if scale == 'radians' else 360
    angle_180 = angle_360 / 2
    angle_90 = angle_360 / 4

    # Get offset from 'from_' to MVSource.
    if from_ == 'kv-detector':
        mv_offset = angle_90 if machine == 'elekta' else -angle_90
    elif from_ == 'kv-source':
        mv_offset = angle_90 if machine == 'varian' else -angle_90
    elif from_ == 'mv-detector':
        mv_offset = 180
    elif from_ == 'mv-source':
        mv_offset = 0
    else:
        raise ValueError(f"Unrecognised position 'from_={from_}'")

    # Get offset from MVSource to 'to'.
    if to == 'kv-detector':
        to_offset = angle_90 if machine == 'varian' else -angle_90
    elif to == 'kv-source':
        to_offset = angle_90 if machine == 'elekta' else -angle_90
    elif to == 'mv-detector':
        to_offset = angle_180
    elif to == 'mv-source':
        to_offset = 0
    else:
        raise ValueError(f"Unrecognised position 'to={to}'")

    # Convert angles.
    offset = mv_offset + to_offset
    angles = [float(np.round((a + offset) % angle_360, decimals=3)) for a in angles]
    if was_single:
        return angles[0]
    else:
        return angles

# Reverses angles (e.g. for converting from source to detector or vice versa).
def reverse_angles(
    angles: float | List[float],
    ) -> List[float]:
    angles, was_single = arg_to_list(angles, (int, float), return_matched=True)
    angles = [float(np.round((360 - a) % 360, decimals=3)) for a in angles]
    if was_single:
        return angles[0]
    else:
        return angles

def plot_treatment_image(
    data: Image2D,
    info: pd.Series,
    alpha_label: float = 0.3,
    ax: mpl.axes.Axes | None = None,
    hist_eq: bool = False,
    labels: BatchLabelImage2D | None = None,
    invert: bool = False,
    markerless_pp: bool = False,
    points: BatchPoints2D | None = None,
    title_fontsize: float = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
    ) -> None:
    if ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False
    if invert:
        data = data.max() - data
    if hist_eq:
        data = hist_eq_fn(data)
    elif markerless_pp:
        data = np.log(data + 1)
        data = data.max() - data
        data = minmax(data)

    det_size = info['det-size']
    aspect = det_size[1] / det_size[0]
    det_spacing = info['det-spacing']
    angle = info.get('kv-source-angle')
    title = f"Treatment image"
    if angle is not None:
        title += f" (kV source. angle={angle:.1f}°)"
    title += f"\nFile: {info.get('filename', 'N/A')}"

    plot_slice(data, ax=ax, aspect=aspect, labels=labels, points=points, labels_alpha=alpha_label,
               title=title, title_fontsize=title_fontsize, x_label=f"LR [{det_spacing[0]}mm]", y_label=f"SI [{det_spacing[1]}mm]",
               vmin=vmin, vmax=vmax, **kwargs)

    if show:
        plt.show()

def plot_drr(
    data: Image2D,
    info: pd.Series,
    ax: mpl.axes.Axes | None = None,
    hist_eq: bool = True,
    normalise: bool = False,
    title_fontsize: float = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ) -> None:
    if ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False

    if normalise:
        data = (data - data.min()) / (data.max() - data.min())
    if hist_eq:
        data = hist_eq_fn(data)

    det_size = info['det-size']
    aspect = det_size[1] / det_size[0]
    det_spacing = info['det-spacing']
    angle = info.get('kv-source-angle')
    title = "DRR image" + f" (kV source. angle={angle:.1f}°)" if angle is not None else ''

    plot_slice(data, ax=ax, aspect=aspect, title=title, title_fontsize=title_fontsize,
               x_label=f"LR [{det_spacing[0]}mm]", y_label=f"SI [{det_spacing[1]}mm]",
               vmin=vmin, vmax=vmax)

    if show:
        plt.show()

def plot_breath(
    info: pd.Series,
    axs: List[mpl.axes.Axes] | None = None,
    all_info: pd.DataFrame | None = None,
    title_fontsize: float = 10,
    ) -> None:
    if axs is None:
        _, axs = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
        show = True
    else:
        assert len(axs) == 2, f"Expected 2 axes for breath plot, got {len(axs)}."
        show = False

    cb_palette = sns.color_palette('colorblind')
    if all_info is not None:
        id_key = next((k for k in ('filepath', 'filename') if k in info.index), None)
        other_info = all_info[all_info[id_key] != info[id_key]] if id_key else all_info
        other_angles = np.deg2rad(other_info['kv-source-angle'].values)
        other_amps = np.array(other_info['MMAmplitude0'].values)
        other_phases = np.array(other_info['MMPhase0'].values)
        base_color = cb_palette[1]
        n = len(other_angles)
        colors = [
            tuple(np.clip(np.array(base_color) * (0.5 + 0.5 * (i / max(n - 1, 1))), 0, 1))
            for i in range(n)
        ]
        for i in range(n):
            axs[0].scatter(other_angles[i], other_amps[i], color=colors[i], s=30, alpha=0.7, edgecolor='none')
            axs[1].scatter(other_angles[i], other_phases[i], color=colors[i], s=30, alpha=0.7, edgecolor='none')

    angle = np.deg2rad(info['kv-source-angle'])
    amp = info['MMAmplitude0']
    phase = info['MMPhase0']
    axs[0].scatter(angle, amp, color=cb_palette[0], s=60, label='Current frame', zorder=10, edgecolor='black')
    axs[1].scatter(angle, phase, color=cb_palette[0], s=60, label='Current frame', zorder=10, edgecolor='black')

    all_amps = [amp] + (list(other_amps) if all_info is not None else [])
    all_phases = [phase] + (list(other_phases) if all_info is not None else [])

    # Some frames have corrupted/missing breathing metadata (NaN) - exclude them from the
    # axis range, otherwise matplotlib raises 'Axis limits cannot be NaN or Inf'.
    valid_amps = np.asarray(all_amps, dtype=float)
    valid_amps = valid_amps[np.isfinite(valid_amps)]
    amp_min, amp_max = (valid_amps.min(), valid_amps.max()) if len(valid_amps) > 0 else (0.0, 1.0)
    valid_phases = np.asarray(all_phases, dtype=float)
    valid_phases = valid_phases[np.isfinite(valid_phases)]
    phase_min, phase_max = (valid_phases.min(), valid_phases.max()) if len(valid_phases) > 0 else (0.0, 1.0)

    axs[0].set_title(f"Breathing amplitude\n[{amp_min:.2f} - {amp_max:.2f}mm]", fontsize=title_fontsize)
    axs[0].set_rlim(amp_min, amp_max)
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)
    axs[0].set_yticklabels([])
    axs[1].set_title(f"Breathing phase\n[{phase_min:.2f} - {phase_max:.2f}°]", fontsize=title_fontsize)
    axs[1].set_rlim(phase_min, phase_max)
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    axs[1].set_yticklabels([])

    if show:
        plt.show()

def plot_projection(
    data: Image2D,
    info: pd.Series,
    all_info: pd.DataFrame | None = None,
    drr: Image2D | None = None,
    labels: BatchLabelImage2D | None = None,
    points: BatchPoints2D | None = None,
    alpha_label: float = 0.3,
    figsize: tuple | None = None,
    hist_eq: bool = False,
    markerless_pp: bool = False,
    invert: bool = False,
    return_image: bool = False,
    title_fontsize: float = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs,
    ) -> None:
    import io
    from PIL import Image as PILImage

    has_breath = 'MMAmplitude0' in info.index and 'MMPhase0' in info.index
    if figsize is None:
        figsize_w = 6 * (1 + (1 if drr is not None else 0) + (1 if has_breath else 0))
        figsize = (figsize_w, 4)
    fig = plt.figure(figsize=figsize)
    width_ratios = [2] + ([2] if drr is not None else []) + ([1, 1] if has_breath else [])
    gs = mpl.gridspec.GridSpec(1, len(width_ratios), width_ratios=width_ratios)

    col = 0
    image_ax = fig.add_subplot(gs[0, col])
    plot_treatment_image(data, info, ax=image_ax, labels=labels, points=points, alpha_label=alpha_label,
                         hist_eq=hist_eq, markerless_pp=markerless_pp, invert=invert, title_fontsize=title_fontsize,
                         vmin=vmin, vmax=vmax, **kwargs)
    image_ax.set_aspect('auto')
    col += 1

    if drr is not None:
        drr_ax = fig.add_subplot(gs[0, col], sharex=image_ax, sharey=image_ax)
        plot_drr(drr, info, ax=drr_ax, hist_eq=hist_eq, title_fontsize=title_fontsize, vmin=vmin, vmax=vmax)
        drr_ax.set_aspect('auto')
        col += 1

    if has_breath:
        amp_ax = fig.add_subplot(gs[0, col], polar=True)
        phase_ax = fig.add_subplot(gs[0, col + 1], polar=True)
        plot_breath(info, all_info=all_info, axs=[amp_ax, phase_ax], title_fontsize=title_fontsize)

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf)
    else:
        plt.show()

def disk_mask(
    size: Size2D,
    points: Point2D | Points2D,
    radius: int = 1
    ) -> LabelImage2D:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]

    # centres as array of shape (N, 2) -> y0s, x0s each shape (N,)
    points = to_numpy(points)
    y0s, x0s = points[:, 0], points[:, 1]

    # broadcast: (N, 1, 1) vs (H, W) -> (N, H, W)
    dist_sq = (xx[None, :, :] - x0s[:, None, None])**2 + (yy[None, :, :] - y0s[:, None, None])**2
    mask = (dist_sq <= radius**2).any(axis=0)
    return mask

def spherical_mask(
    size: Size3D,
    points: Point3D | Points3D,
    radius: int = 1
    ) -> LabelImage3D:
    h, w, d = size
    zz, yy, xx = np.mgrid[0:h, 0:w, 0:d]

    # centres as array of shape (N, 3) -> z0s, y0s, x0s each shape (N,)
    points = to_numpy(points)
    z0s, y0s, x0s = points[:, 0], points[:, 1], points[:, 2]

    # broadcast: (N, 1, 1, 1) vs (H, W, D) -> (N, H, W, D)
    dist_sq = ((xx[None] - x0s[:, None, None, None])**2 +
               (yy[None] - y0s[:, None, None, None])**2 +
               (zz[None] - z0s[:, None, None, None])**2)
    mask = (dist_sq <= radius**2).any(axis=0)
    return mask

# Scales the affine to account for different spacing at the isocentre in comparison
# to the detector plane.
def affine_at_isocentre(
    affine: AffineMatrix2D,
    sid: float,
    sdd: float,
    ) -> AffineMatrix2D:
    scale = sid / sdd
    affine_scaled = np.copy(affine)
    affine_scaled[:2, :2] *= scale
    return affine_scaled

def blur_image(
    image: Image2D | Image3D,
    dilate: int = 0,    # Dilates the mask.
    mask: LabelImage2D | LabelImage3D | None = None,
    points: Point2D | Points2D | None = None,
    **kwargs,
    ) -> Image2D | Image3D:
    assert mask is not None or points is not None, "Either mask or points must be provided."
    image = np.asarray(image)
    if mask is None:
        if image.ndim == 3:
            mask = spherical_mask(image.shape, points=points, **kwargs)
        else:
            mask = disk_mask(image.shape, points=points, **kwargs)
    if dilate > 0:
        mask = binary_dilation(mask, iterations=dilate)
    # Perform region-fill - just blurring inwards really.
    image = inpaint_biharmonic(image, mask, split_into_regions=False)
    # Add poisson noise, with distribution centred on the blurred pixel intensities.
    # Poisson noise only makes sense with 2D projection images.
    if image.ndim == 2:
        POISSON_SCALE = 65535
        orig_min = image.min()
        orig_max = image.max()
        # Rescale to integer range so Poisson draws are meaningful.
        noise = (image.astype(np.float64) - orig_min) / (orig_max - orig_min) * POISSON_SCALE
        for _ in range(20):
            noise = np.clip(np.random.poisson(noise), 0, POISSON_SCALE).astype(np.float64)
        # Rescale back to original range.
        noise = noise / POISSON_SCALE * (orig_max - orig_min) + orig_min
        image = np.where(mask, noise, image.astype(np.float64))
    return image

def load_shroud(
    path: str,
    ) -> np.ndarray:
    from dicomset.utils import load_numpy
    return load_numpy(path)

def plot_shroud(
    shroud: np.ndarray,
    ax: mpl.axes.Axes | None = None,
    title: str | None = None,
    ) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))
        show = True
    else:
        show = False
    ax.imshow(shroud.T, cmap='gray', origin='lower', aspect='auto')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Column (px)')
    if title is not None:
        ax.set_title(title)
    if show:
        plt.tight_layout()
        plt.show()

def stats_by_bin(
    angles: List[float] | np.ndarray,
    data: List[float] | np.ndarray,
    bin_width: float = 360 / 10,
    bin_offset: float = (360 / 10) / 2,
    ) -> Tuple[List[str], np.ndarray]:
    angles = np.asarray(angles, dtype=float)
    data   = np.asarray(data,   dtype=float)
    n_bins = int(360 // bin_width)

    # Map each angle to a bin index.
    shifted   = (angles + bin_offset) % 360
    bin_idxs  = (shifted // bin_width).astype(int)

    # Build label strings "[lo, hi)" for each possible bin index.
    def _label(idx: int) -> str:
        lo = int(idx * bin_width - bin_offset)
        hi = int(idx * bin_width + bin_offset)
        return f"[{lo}, {hi})"

    # Aggregate stats per bin, sorted by the numeric lower edge.
    records = []
    for idx in range(n_bins):
        mask   = bin_idxs == idx
        values = data[mask]
        lo     = idx * bin_width - bin_offset
        if values.size == 0:
            continue
        records.append((lo, _label(idx),
                        float(np.mean(values)),
                        float(np.median(values)),
                        float(np.std(values))))

    records.sort(key=lambda r: r[0])
    bins  = [r[1] for r in records]
    stats = np.array([[r[2], r[3], r[4]] for r in records], dtype=float)
    return bins, stats
