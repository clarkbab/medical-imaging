from augmed.typing import *
from augmed.utils import to_tensor, to_numpy
from dicomset.utils import logger, hist_eq as hist_eq_fn, plot_slice
from dicomset.utils.args import arg_to_list
from dicomset.utils.geometry import affine_origin, affine_spacing, create_affine, fov_centre
from CTorch.utils.geometry import CircGeom3D
from CTorch.projector.projector_interface import Projector
from diffdrr.data import read
from diffdrr.drr import DRR
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydicom as dcm
import seaborn as sns
import subprocess
import tempfile
import torch
import torchio as tio
from tqdm import tqdm
from typing import *

from mymi.utils.ct import fill_ct_background, has_ct_background
from mymi.typing import *
from mymi.utils.io import sitk_load_volume, sitk_save_volume

def project_diffdrr(
    volume: Volume,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_detector_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    couch_shift: Point3D = (0.0, 0.0, 0.0),
    labels: LabelVolumeBatch | None = None,
    n_angles_batch: int | None = 20,
    patch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = False,
    ) -> SliceBatch | Tuple[SliceBatch, LabelSliceBatch]:
    if verbose:
        logger.log_method('Creating DiffDRR projections')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    volume = to_tensor(volume, device=device, dtype=torch.float32)
    affine = to_tensor(affine, device=volume.device, dtype=torch.float32)
    treatment_iso = to_tensor(treatment_iso, device=volume.device, dtype=torch.float32)
    det_spacing = to_tensor(det_spacing, device=volume.device, dtype=torch.float32)
    det_offset = to_tensor(det_offset, device=volume.device, dtype=torch.float32)
    couch_shift = to_tensor(couch_shift, device=volume.device, dtype=torch.float32)
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
    kv_detector_angles = arg_to_list(kv_detector_angles, float)
    # TODO: Check which voxel is the treatment isocentre, if it's too
    # far from centre the DRR might look whacky.
    # treatment_iso_vox = point_to_image_coords(treatment_iso, affine)
    # print(f"Treatment isocentre (image coords): {treatment_iso_vox}, should be near\
    #     image centre voxel {tuple(s//2 for s in volume.shape)}.")

    # Move the treatment isocentre to the origin.
    print(affine.device)
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    print(origin.device)
    print(treatment_iso.device)
    print(couch_shift.device)
    origin = origin - treatment_iso - couch_shift
    affine = create_affine(spacing, origin)

    # Create torchio objects.
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)
    volume = volume.unsqueeze(0) if len(volume.shape) == 3 else volume
    tio_volume = tio.ScalarImage(tensor=volume, affine=affine)

    # Split angles into batches if requested.
    if n_angles_batch is None:
        batches = [kv_detector_angles]
    else:
        batches = [kv_detector_angles[i:i + n_angles_batch] for i in range(0, len(kv_detector_angles), n_angles_batch)]

    # Need to iterate over labels!
    # See: https://github.com/eigenvivek/DiffDRR/issues/409
    n_label_iter = 1 if labels is None else len(labels)
    labels_proj = []

    for i in tqdm(range(n_label_iter), desc="Creating label projections..."):
        if labels is not None:
            if isinstance(labels, np.ndarray):
                label_map = torch.from_numpy(labels[i]).unsqueeze(0)
            else:
                label_map = labels[i].unsqueeze(0)
            label_map = tio.LabelMap(tensor=label_map, affine=affine)
        else:
            label_map = None

        subject = read(tio_volume, center_volume=False, labelmap=label_map)

        # Create DRR object.
        drr = DRR(
            subject,
            sdd=sdd,
            width=int(det_size[0]),
            height=int(det_size[1]),
            patch_size=patch_size,
            delx=det_spacing[0],
            dely=det_spacing[1],
            x0=det_offset[0],
            y0=det_offset[1],
            reverse_x_axis=False,
        ).to(volume.device)

        # Project in angle batches.
        volume_proj_chunks = []
        label_proj_chunks = []

        for angles in tqdm(batches, desc="DiffDRR angle batches", leave=False):
            rot = torch.tensor([
                [a, 0.0, 0.0] for a in angles
            ], device=volume.device, dtype=torch.float32)
            trans = torch.tensor([
                [0, sid, 0.0] for _ in angles
            ], device=volume.device, dtype=torch.float32)

            with torch.no_grad():
                mask_to_channels = True if labels is not None else False
                proj_data = drr(rot, trans, parameterization="euler_angles", convention="ZYX", degrees=True, mask_to_channels=mask_to_channels)

            # Correct x/y axes.
            proj_data = proj_data.moveaxis(2, 3)

            # Move to CPU immediately.
            volume_proj_chunks.append(proj_data[:, 0].cpu().numpy())
            if labels is not None:
                lp = proj_data[:, 1].cpu().numpy()
                lp[lp != 0] = 1
                lp = lp.astype(np.bool_)
                label_proj_chunks.append(lp)

            # Free GPU memory before next angle batch.
            del proj_data, rot, trans
            torch.cuda.empty_cache()

        # Concatenate angle batches.
        volume_proj = np.concatenate(volume_proj_chunks, axis=0)
        del volume_proj_chunks
        if labels is not None:
            labels_proj.append(np.concatenate(label_proj_chunks, axis=0))
            del label_proj_chunks

        # Free GPU memory before next label iteration.
        del drr, subject
        if label_map is not None:
            del label_map
        torch.cuda.empty_cache()

        if progress_callback:
            progress_callback(i + 1, n_label_iter)

    volume_proj = np.flip(volume_proj, axis=-1)
    if labels is not None:
        labels_proj = np.flip(labels_proj, axis=-1)

    if labels is not None:
        labels_proj = np.stack(labels_proj, axis=1)
        return volume_proj, labels_proj
    else:
        return volume_proj

# I think this actually takes kv detector angles, see contour alignment tool behaviour.
def project_igt(
    volume: Volume,
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
    kv_detector_angles: List[float],
    # For half-fan CBCT, the patient is centred laterally before the CBCT image is acquired to
    # avoid collision.
    couch_shift: Point3D = (0.0, 0.0, 0.0),
    dirpath: DirPath | None = None,
    labels: LabelVolumeBatch | None = None,
    recreate: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = False,
    ) -> SliceBatch | Tuple[SliceBatch, LabelSliceBatchChannel]:
    # We need a folder to save the intermediate (.mha, .csv, .xml) files. 
    if verbose:
        logger.log_method('Creating IGT projections')
    if dirpath is None:
        dirpath = tempfile.gettempdir()
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
        assert labels.shape[1:] == volume.shape, f"Expected labels to have same spatial shape \
            as volume, got {labels.shape[1:]} but expected {volume.shape}"

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if has_ct_background(volume):
        volume = fill_ct_background(volume)

    # Does IGT use a counter-clockwise positive, (CCW+) angular convention?
    # Needed this to look right.
    kv_detector_angles = reverse_angles(kv_detector_angles)

    # Flip data along z axis - required by IGT.
    # This was something that just worked.
    # I think this is because the .xim files used a CCW+ angular convention,
    # and one way to achieve this in a CW+ convention is to flip the z-axis.
    volume = np.flip(volume, axis=2)
    if labels is not None:
        labels = np.flip(labels, axis=-1)

    # Move the volume to the treatment isocentre.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
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
    if labels is not None:
        labels = np.moveaxis(labels, -2, -1)

    # If existing angles file doesn't match, then recreate.
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    angles_filepath = os.path.join(dirpath, 'angles.csv')
    if os.path.exists(angles_filepath):
        df = pd.read_csv(angles_filepath, header=None)
        if len(df[0].tolist()) != len(kv_detector_angles) or not np.allclose(df[0].tolist(), kv_detector_angles):
            print(f"Existing angles file {angles_filepath} doesn't match provided angles, recreating projections.")
            recreate = True

    # Write angles to CSV.
    if not os.path.exists(angles_filepath) or recreate:
        with open(angles_filepath, 'w') as f:
            for a in kv_detector_angles:
                f.write(f"{a}\n")
    
    # Create geometry file.
    geometry_filepath = os.path.join(dirpath, 'geometry.xml')
    if not os.path.exists(geometry_filepath) or recreate:
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

    # Create the CT and ROI MHA files.
    ct_filepath = os.path.join(dirpath, 'CT.mha')
    if not os.path.exists(ct_filepath) or recreate:
        sitk_save_volume(volume, affine, ct_filepath)
    if labels is not None:
        for i in range(len(labels)):
            roi_filepath = os.path.join(dirpath, f'ROI_{i}.mha')
            if not os.path.exists(roi_filepath) or recreate:
                sitk_save_volume(labels[i], affine, roi_filepath)
    
    # Create the CT forward projections.
    n_labels = len(labels) if labels is not None else 0
    total_steps = 1 + n_labels
    ct_fp_filepath = os.path.join(dirpath, 'CT_FP.mha')
    if not os.path.exists(ct_fp_filepath) or recreate:
        cmd = [
            r"D:\Brett\code\IGT\IGTExecutables\VS2013_x64\rtkforwardprojections.exe",
            "-i", ct_filepath,
            "-o", ct_fp_filepath,
            "-g", geometry_filepath,
            "--fp", "Joseph",
            "--dimension", f"{det_size[0]},{det_size[1]}",
            "--spacing", str(det_spacing[0]),
        ]
        print(cmd)
        subprocess.run(cmd)
    if progress_callback:
        progress_callback(1, total_steps)

    # Forward project the ROIs.
    if labels is not None:
        for i in range(len(labels)):
            roi_filepath = os.path.join(dirpath, f'ROI_{i}.mha')
            roi_fp_filepath = os.path.join(dirpath, f'ROI_{i}_FP.mha')
            if not os.path.exists(roi_fp_filepath) or recreate:
                cmd = [
                    r"D:\Brett\code\IGT\IGTExecutables\VS2013_x64\rtkforwardprojections.exe",
                    "-i", roi_filepath,
                    "-o", roi_fp_filepath,
                    "-g", geometry_filepath,
                    "--fp", "Joseph",
                    "--dimension", f"{det_size[0]},{det_size[1]}",
                    "--spacing", str(det_spacing[0]),
                ]
                print(cmd)
                subprocess.run(cmd)
            if progress_callback:
                progress_callback(2 + i, total_steps)

    # Load results.
    ct_fp, _ = sitk_load_volume(ct_fp_filepath)
    ct_fp = np.moveaxis(ct_fp, -1, 0)
    ct_fp = np.flip(ct_fp, axis=-1)
    if labels is not None:
        labels_fp = np.zeros((len(kv_detector_angles), len(labels), *det_size), dtype=np.bool_)
        for i in range(len(labels)):
            roi_fp_filepath = os.path.join(dirpath, f'ROI_{i}_FP.mha')
            roi_fp, _ = sitk_load_volume(roi_fp_filepath)
            roi_fp = np.moveaxis(roi_fp, -1, 0)
            # Binarise the ROI projections.
            roi_fp[roi_fp != 0] = 1
            roi_fp = roi_fp.astype(np.bool_)
            labels_fp[:, i] = roi_fp
            labels_fp = np.flip(labels_fp, axis=-1)
        return ct_fp, labels_fp
    else:
        return ct_fp

def project_ctorch(
    volume: Image3D,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_detector_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    couch_shift: Point3D = (0.0, 0.0, 0.0),
    labels: BatchLabelImage3D | None = None,
    n_angles_batch: int | None = 20,
    threshold_labels: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    verbose: bool = False,
    ) -> BatchImage2D | Tuple[BatchImage2D, BatchChannelLabelImage2D]:
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
    kv_detector_angles = arg_to_list(kv_detector_angles, float)
    kv_detector_angles = to_numpy(kv_detector_angles)
    if verbose:
        logger.log_method('Creating CTorch projections')
    # TODO: Check which voxel is the treatment isocentre, if it's too
    # far from centre the DRR might look whacky.
    # treatment_iso_vox = point_to_image_coords(treatment_iso, affine)
    # print(f"Treatment isocentre (image coords): {treatment_iso_vox}, should be near\
    #     image centre voxel {tuple(s//2 for s in volume.shape)}.")

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if has_ct_background(volume):
        volume = fill_ct_background(volume)

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
        volume_tensor = torch.concat([volume_tensor.unsqueeze(0), labels_tensor], axis=0).unsqueeze(0).cuda()
    else:
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0).cuda()
    # Flip z, transpose x/y/z.
    volume_tensor = volume_tensor.flip(4).permute((0, 1, 4, 3, 2))

    # Split angles into batches if requested.
    if not kv_detector_angles:
        raise ValueError('kv_detector_angles is empty — cannot project with zero angles')
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

        projector = Projector(geom, 'proj', 'SF', 'forward')
        with torch.no_grad():
            proj_data = projector(volume_tensor)

        vp = proj_data[0, 0].permute((0, 2, 1)).cpu()
        volume_proj_chunks.append(vp)

        if labels is not None:
            lp = proj_data[0, 1:].permute((1, 0, 3, 2)).cpu()
            if threshold_labels:
                lp[lp >= 1] = 1
                lp[lp < 0] = 0
                lp = lp.type(torch.bool)
            labels_proj_chunks.append(lp)

        if progress_callback:
            progress_callback(i + 1, n_batches)

        # Free GPU memory before next batch.
        del proj_data, projector, geom
        torch.cuda.empty_cache()

    # Concatenate batches along the angles dimension.
    volume_proj = torch.cat(volume_proj_chunks, dim=0)
    del volume_proj_chunks
    volume_proj = torch.flip(volume_proj, dims=[-1])

    if labels is not None:
        labels_proj = torch.cat(labels_proj_chunks, dim=0)
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
    hist_eq: bool = True,
    labels: BatchLabelImage2D | None = None,
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
    angle = info.get('kv-detector-angle')
    title = "Treatment image" + f" (kV det. angle={angle:.1f}°)" if angle is not None else ''

    plot_slice(data, ax=ax, aspect=aspect, labels=labels, alpha=alpha_label,
               title=title, title_fontsize=title_fontsize, x_label=f"LR [{det_spacing[0]}mm]", y_label=f"SI [{det_spacing[1]}mm]",
               vmin=vmin, vmax=vmax)

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
    angle = info.get('kv-detector-angle')
    title = "DRR image" + f" (kV det. angle={angle:.1f}°)" if angle is not None else ''

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
        other_angles = np.deg2rad(other_info['kv-detector-angle'].values)
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

    angle = np.deg2rad(info['kv-detector-angle'])
    amp = info['MMAmplitude0']
    phase = info['MMPhase0']
    axs[0].scatter(angle, amp, color=cb_palette[0], s=60, label='Current frame', zorder=10, edgecolor='black')
    axs[1].scatter(angle, phase, color=cb_palette[0], s=60, label='Current frame', zorder=10, edgecolor='black')

    all_amps = [amp] + (list(other_amps) if all_info is not None else [])
    all_phases = [phase] + (list(other_phases) if all_info is not None else [])
    axs[0].set_title(f"Breathing amplitude\n[{np.min(all_amps):.2f} - {np.max(all_amps):.2f}mm]", fontsize=title_fontsize)
    axs[0].set_rlim(min(all_amps), max(all_amps))
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)
    axs[0].set_yticklabels([])
    axs[1].set_title(f"Breathing phase\n[{np.min(all_phases):.2f} - {np.max(all_phases):.2f}°]", fontsize=title_fontsize)
    axs[1].set_rlim(min(all_phases), max(all_phases))
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
    alpha_label: float = 0.3,
    figsize: tuple | None = None,
    hist_eq: bool = True,
    normalise: bool = False,
    return_image: bool = False,
    title_fontsize: float = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
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
    plot_treatment_image(data, info, ax=image_ax, labels=labels, alpha_label=alpha_label,
                         hist_eq=hist_eq, normalise=normalise, title_fontsize=title_fontsize,
                         vmin=vmin, vmax=vmax)
    image_ax.set_aspect('auto')
    col += 1

    if drr is not None:
        drr_ax = fig.add_subplot(gs[0, col], sharex=image_ax, sharey=image_ax)
        plot_drr(drr, info, ax=drr_ax, hist_eq=hist_eq, normalise=normalise, title_fontsize=title_fontsize, vmin=vmin, vmax=vmax)
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
