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

from mymi import logging
from mymi.typing import *

from .affine import create_affine, affine_origin, affine_spacing
from .args import arg_to_list
from .io import sitk_load_volume, sitk_save_volume
from .points import point_to_image_coords

def convert_angles(
    angles: List[float],
    from_: Literal['kv-detector', 'kv-source', 'mv-detector', 'mv-source'],
    to: Literal['kv-detector', 'kv-source', 'mv-detector', 'mv-source'],
    machine: Literal['elekta', 'varian'],
    ) -> List[float]:
    # Get offset from 'from_' to MVSource.
    if from_ == 'kv-detector':
        mv_offset = 90 if machine == 'elekta' else -90
    elif from_ == 'kv-source':
        mv_offset = 90 if machine == 'varian' else -90
    elif from_ == 'mv-detector':
        mv_offset = 180
    elif from_ == 'mv-source':
        mv_offset = 0
    else:
        raise ValueError(f"Unrecognised position 'from_={from_}'")

    # Get offset from MVSource to 'to'.
    if to == 'kv-detector':
        to_offset = 90 if machine == 'varian' else -90
    elif to == 'kv-source':
        to_offset = 90 if machine == 'elekta' else -90
    elif to == 'mv-detector':
        to_offset = 180
    elif to == 'mv-source':
        to_offset = 0
    else:
        raise ValueError(f"Unrecognised position 'to={to}'")

    # Convert angles.
    offset = mv_offset + to_offset
    angles = [float(np.round((a + offset) % 360, decimals=3)) for a in angles]
    return angles

# Where can this method go wrong?
# 1. When the 'kv_source_angles' don't use the clockwise positive, (CW+, looking foot -> head) convention.
#    You'll know this is the case when kV detector angles 90/270 look right, but 0/180
#    look reversed. This is because MV gantry 0/180 are the same in both coordinate systems.
#    I found that .xim files used a counter-clockwise positive, (CCW+) convention and had to be corrected
#    before using this code.
def create_projections(
    volume: Volume,
    affine: Affine,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: float | List[float],    # We pass these instead of MV gantry, as the relative position changes between machines.
    labels: LabelVolumeBatch | None = None,
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
    logging.info(f"Creating projections (device={device}) with geometry: treatment_iso={treatment_iso}, sid={sid}, sdd={sdd}, det_size={det_size}, det_spacing={det_spacing}, det_offset={det_offset}, kv_source_angles={kv_source_angles}")

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

    # Need to iterate over labels!
    # See: https://github.com/eigenvivek/DiffDRR/issues/409
    n_iter = 1 if labels is None else len(labels)
    labels_proj = []
    for i in tqdm(range(n_iter), desc="Creating label projections..."):
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

        # Create projections
        rot = torch.tensor([
            [a, 0.0, 0.0] for a in kv_source_angles
        ], device=device, dtype=torch.float32)
        trans = torch.tensor([
            [0, sid, 0.0] for _ in kv_source_angles
        ], device=device, dtype=torch.float32)
        with torch.no_grad():
            mask_to_channels = True if labels is not None else False
            proj_data = drr(rot, trans, parameterization="euler_angles", convention="ZYX", degrees=True, mask_to_channels=mask_to_channels)

        # Correct x/y axes.
        proj_data = proj_data.moveaxis(2, 3)
        
        # Split into DRR and labels.
        volume_proj = proj_data[:, 0].cpu().numpy()
        if labels is not None:
            lp = proj_data[:, 1].cpu().numpy()
            lp[lp != 0] = 1
            lp = lp.astype(np.bool_)
            labels_proj.append(lp)

    if labels is not None:
        labels_proj = np.stack(labels_proj, axis=1)
        return volume_proj, labels_proj
    else:
        return volume_proj

def create_igt_projections(
    ct_volume: CtVolume,
    affine: Affine,
    treatment_iso: Point3D,
    sid: float,
    sdd: float,
    det_size: Size2D,
    det_spacing: Spacing2D,
    det_offset: Point2D,
    kv_source_angles: List[float],
    dirpath: DirPath | None = None,
    labels: LabelVolumeBatch | None = None,
    recreate: bool = False,
    ) -> SliceBatch | Tuple[SliceBatch, LabelSliceBatchChannel]:
    # We need a folder to save the intermediate (.mha, .csv, .xml) files. 
    if dirpath is None:
        dirpath = tempfile.gettempdir()
    if labels is not None:
        assert len(labels.shape) == 4, f"Expected labels to have shape (C, X, Y, Z), got {labels.shape}."
        assert labels.shape[1:] == ct_volume.shape, f"Expected labels to have same spatial shape \
            as ct_volume, got {labels.shape[1:]} but expected {ct_volume.shape}"

    # Does IGT use a counter-clockwise positive, (CCW+) angular convention?
    # Needed this to look right.
    kv_source_angles = reverse_angles(kv_source_angles)

    # Flip data along z axis - required by IGT.
    # This was something that just worked.
    # I think this is because the .xim files used a CCW+ angular convention,
    # and one way to achieve this in a CW+ convention is to flip the z-axis.
    ct_volume = np.flip(ct_volume, axis=2)
    if labels is not None:
        labels = np.flip(labels, axis=-1)

    # Move the treatment isocentre to the origin.
    spacing = affine_spacing(affine)
    origin = affine_origin(affine)
    origin = np.array(origin)
    origin[0] = origin[0] - treatment_iso[0]
    origin[1] = origin[1] - treatment_iso[1]
    # Need to adjust the z origin because of flipped axis.
    max_z_loc = origin[2] + (ct_volume.shape[2] - 1) * spacing[2]
    origin[2] = treatment_iso[2] - max_z_loc

    # Reorder axes to (x, z, y) - required by IGT.
    # Is this IEC geometry?
    ct_volume = np.moveaxis(ct_volume, 1, 2)
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
            r"E:\Brett\code\IGT\IGTExecutables\VS2013_x64\igtsimulatedgeometry.exe",
            "-i", angles_filepath,
            "--sid", str(sid),
            "--sdd", str(sdd),
            "--proj_iso_x", str(det_offset[0]),
            "-o", geometry_filepath,
        ]
        print(cmd)
        subprocess.run(cmd)

    # Create the CT and ROI MHA files.
    ct_filepath = os.path.join(dirpath, 'CT.mha')
    if not os.path.exists(ct_filepath) or recreate:
        sitk_save_volume(ct_volume, affine, ct_filepath)
    if labels is not None:
        for i in range(len(labels)):
            roi_filepath = os.path.join(dirpath, f'ROI_{i}.mha')
            if not os.path.exists(roi_filepath) or recreate:
                sitk_save_volume(labels[i], affine, roi_filepath)
    
    # Create the CT forward projections.
    ct_fp_filepath = os.path.join(dirpath, 'CT_FP.mha')
    if not os.path.exists(ct_fp_filepath) or recreate:
        cmd = [
            r"E:\Brett\code\IGT\IGTExecutables\VS2013_x64\rtkforwardprojections.exe",
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
                    r"E:\Brett\code\IGT\IGTExecutables\VS2013_x64\rtkforwardprojections.exe",
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

def reverse_angles(
    angles: float | List[float],
    ) -> float | List[float]:
    angles, was_single = arg_to_list(angles, (int, float), return_matched=True)
    angles = [float(np.round((360 - a) % 360, decimals=3)) for a in angles]
    if was_single:
        return angles[0]
    else:
        return angles

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

    kv_source_angles = []
    for i, f in enumerate(ds.ExposureSequence):
        frame = int(f.ReferencedFrameNumber)
        assert frame == i + 1
        angle = float(getattr(f, "GantryAngle", np.nan))
        kv_source_angles.append(angle)
    info['kv-source-angles'] = kv_source_angles

    # Load pixel data.
    data = ds.pixel_array
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    data = data * slope + intercept

    # Transpose x/y axes.
    data = np.moveaxis(data, 1, 2)
    info['det-size'] = data.shape[1:]

    return data, info
