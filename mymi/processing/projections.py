from augmed.typing import *
from augmed.utils import to_tensor
from CTorch.utils.geometry import CircGeom3D
from CTorch.projector.projector_interface import Projector
from diffdrr.data import read
from diffdrr.drr import DRR
import numpy as np
import os
import pandas as pd
import pydicom as dcm
import subprocess
import tempfile
import torch
import torchio as tio
from tqdm import tqdm
from typing import *

from mymi.geometry import fov_centre
from mymi import logging
from mymi.processing.ct import fill_ct_background, has_ct_background
from mymi.typing import *
from mymi.utils import (
    arg_to_list, create_affine, affine_origin, affine_spacing, sitk_load_volume, 
    sitk_save_volume, point_to_image_coords, reverse_angles,
)

# Where can this method go wrong?
# 1. When the 'kv_source_angles' don't use the clockwise positive, (CW+, looking foot -> head) convention.
#    You'll know this is the case when kV detector angles 90/270 look right, but 0/180
#    look reversed. This is because MV gantry 0/180 are the same in both coordinate systems.
#    I found that .xim files used a counter-clockwise positive, (CCW+) convention and had to be corrected
#    before using this code.
def create_diffdrr_projections(
    volume: Volume,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    labels: LabelVolumeBatch | None = None,
    n_angles_batch: int | None = 20,
    patch_size: int = 128,
    ) -> SliceBatch | Tuple[SliceBatch, LabelSliceBatch]:
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
    kv_source_angles = arg_to_list(kv_source_angles, float)
    # TODO: Check which voxel is the treatment isocentre, if it's too
    # far from centre the DRR might look whacky.
    # treatment_iso_vox = point_to_image_coords(treatment_iso, affine)
    # print(f"Treatment isocentre (image coords): {treatment_iso_vox}, should be near\
    #     image centre voxel {tuple(s//2 for s in volume.shape)}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.log_args(f"Creating DiffDRR projections (device={device})")

    # Move the treatment isocentre to the origin.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    origin = tuple(np.array(origin) - treatment_iso)
    affine = create_affine(spacing, origin)

    # Create torchio objects.
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume)
    volume = volume.unsqueeze(0) if len(volume.shape) == 3 else volume
    volume = tio.ScalarImage(tensor=volume, affine=affine)

    # Split angles into batches if requested.
    if n_angles_batch is None:
        batches = [kv_source_angles]
    else:
        batches = [kv_source_angles[i:i + n_angles_batch] for i in range(0, len(kv_source_angles), n_angles_batch)]

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

        subject = read(volume, center_volume=False, labelmap=label_map)

        # Create DRR object.
        drr = DRR(
            subject,
            sdd=sdd,
            width=det_size[0],
            height=det_size[1],
            patch_size=patch_size,
            delx=det_spacing[0],
            dely=det_spacing[1],
            x0=det_offset[0],
            y0=det_offset[1],
            reverse_x_axis=False,
        ).to(device)

        # Project in angle batches.
        volume_proj_chunks = []
        label_proj_chunks = []

        for angles in tqdm(batches, desc="DiffDRR angle batches", leave=False):
            rot = torch.tensor([
                [a, 0.0, 0.0] for a in angles
            ], device=device, dtype=torch.float32)
            trans = torch.tensor([
                [0, sid, 0.0] for _ in angles
            ], device=device, dtype=torch.float32)

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

    if labels is not None:
        labels_proj = np.stack(labels_proj, axis=1)
        return volume_proj, labels_proj
    else:
        return volume_proj

def create_igt_projections(
    volume: Volume,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: List[float],
    dirpath: DirPath | None = None,
    labels: LabelVolumeBatch | None = None,
    recreate: bool = True,
    ) -> SliceBatch | Tuple[SliceBatch, LabelSliceBatchChannel]:
    # We need a folder to save the intermediate (.mha, .csv, .xml) files. 
    if dirpath is None:
        dirpath = tempfile.gettempdir()
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
        assert labels.shape[1:] == volume.shape, f"Expected labels to have same spatial shape \
            as volume, got {labels.shape[1:]} but expected {volume.shape}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.log_args(f"Creating IGT projections (device={device})")

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if has_ct_background(volume):
        volume = fill_ct_background(volume)

    # Does IGT use a counter-clockwise positive, (CCW+) angular convention?
    # Needed this to look right.
    kv_source_angles = reverse_angles(kv_source_angles)

    # Flip data along z axis - required by IGT.
    # This was something that just worked.
    # I think this is because the .xim files used a CCW+ angular convention,
    # and one way to achieve this in a CW+ convention is to flip the z-axis.
    volume = np.flip(volume, axis=2)
    if labels is not None:
        labels = np.flip(labels, axis=-1)

    # Move the treatment isocentre to the origin.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    origin = np.array(origin)
    origin[0] = origin[0] - treatment_iso[0]
    origin[1] = origin[1] - treatment_iso[1]
    # Need to adjust the z origin because of flipped axis.
    max_z_loc = origin[2] + (volume.shape[2] - 1) * spacing[2]
    origin[2] = treatment_iso[2] - max_z_loc

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
        if len(df[0].tolist()) != len(kv_source_angles) or not np.allclose(df[0].tolist(), kv_source_angles):
            print(f"Existing angles file {angles_filepath} doesn't match provided angles, recreating projections.")
            recreate = True

    # Write angles to CSV.
    if not os.path.exists(angles_filepath) or recreate:
        with open(angles_filepath, 'w') as f:
            for a in kv_source_angles:
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

    # Load results.
    ct_fp, _ = sitk_load_volume(ct_fp_filepath)
    ct_fp = np.moveaxis(ct_fp, -1, 0)
    if labels is not None:
        labels_fp = np.zeros((len(kv_source_angles), len(labels), *det_size), dtype=np.bool_)
        for i in range(len(labels)):
            roi_fp_filepath = os.path.join(dirpath, f'ROI_{i}_FP.mha')
            roi_fp, _ = sitk_load_volume(roi_fp_filepath)
            roi_fp = np.moveaxis(roi_fp, -1, 0)
            # Binarise the ROI projections.
            roi_fp[roi_fp != 0] = 1
            roi_fp = roi_fp.astype(np.bool_)
            labels_fp[:, i] = roi_fp
        return ct_fp, labels_fp
    else:
        return ct_fp

def create_ctorch_projections(
    volume: Image3D,
    affine: AffineMatrix3D,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    labels: BatchLabelImage3D | None = None,
    n_angles_batch: int | None = 20,
    ) -> BatchImage2D | Tuple[BatchImage2D, BatchChannelLabelImage2D]:
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
    kv_source_angles = arg_to_list(kv_source_angles, float)
    # TODO: Check which voxel is the treatment isocentre, if it's too
    # far from centre the DRR might look whacky.
    # treatment_iso_vox = point_to_image_coords(treatment_iso, affine)
    # print(f"Treatment isocentre (image coords): {treatment_iso_vox}, should be near\
    #     image centre voxel {tuple(s//2 for s in volume.shape)}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.log_args(f"Creating CTorch projections (device={device})")

    # Fill in any synthetic background voxels (e.g. -8000 HU cylinder).
    if has_ct_background(volume):
        volume = fill_ct_background(volume)

    nx, ny, nz = volume.shape
    dx, dy, dz = affine_spacing(affine)
    nu, nv = det_size
    du, dv = det_spacing
    detType = 'flat' #'curve'
    # Convert angles to rad and use CCW+ system.
    kv_source_angles = [np.deg2rad(-a) for a in kv_source_angles]
    SAD, SDD = [sid], [sdd] # source-axis-distance, source-detector-distance
    
    image_centre = fov_centre(volume.shape, affine=affine)
    offset = np.array(image_centre) - treatment_iso
    offset[2] = -offset[2]  # Z-axis is flipped.
    xOfst, yOfst, zOfst = [offset[0]], [offset[1]], [offset[2]] # image center offset
    uOfst, vOfst = [det_offset[0]], [det_offset[1]] # detecor center offset
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
    if n_angles_batch is None:
        batches = [kv_source_angles]
    else:
        batches = [kv_source_angles[i:i + n_angles_batch] for i in range(0, len(kv_source_angles), n_angles_batch)]

    volume_proj_chunks = []
    labels_proj_chunks = [] if labels is not None else None

    for angles in tqdm(batches, desc="CTorch angle batches"):
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
            lp[lp >= 1] = 1
            lp[lp < 0] = 0
            lp = lp.type(torch.bool)
            labels_proj_chunks.append(lp)

        # Free GPU memory before next batch.
        del proj_data, projector, geom
        torch.cuda.empty_cache()

    # Concatenate batches along the angles dimension.
    volume_proj = torch.cat(volume_proj_chunks, dim=0)
    del volume_proj_chunks

    if labels is not None:
        labels_proj = torch.cat(labels_proj_chunks, dim=0)
        del labels_proj_chunks
        return volume_proj, labels_proj
    else:
        return volume_proj
