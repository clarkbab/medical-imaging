from dataclasses import dataclass, field
from datetime import datetime, timedelta
from dicomset.utils import hist_eq as hist_eq_fn, load_csv, load_numpy, logger, save_csv, save_numpy
from dicomset.typing import *
import numpy as np
import os
from pathlib import Path
import pylinac as plc
import struct
from tqdm.auto import tqdm
from typing import Any, Tuple

from mymi.typing import *

from .args import arg_to_list
from .projections import convert_angles, plot_projection, reverse_angles

@dataclass
class XimHistogramInfo:
    number_of_bins: int = 0
    data: np.ndarray | None = None

@dataclass
class XimInfo:
    file_name: str = ""
    file_format_identifier: str = ""
    file_format_version: int = 0
    image_width: int = 0
    image_height: int = 0
    bits_per_pixel: int = 0
    bytes_per_pixel: int = 0
    compression_indicator: int = 0
    histogram: XimHistogramInfo = field(default_factory=XimHistogramInfo)
    properties: dict[str, Any] = field(default_factory=dict)

@dataclass
class HncInfo:
    file_name: str = ""
    file_type: str = ""
    file_length: int = 0
    actual_file_length: int = 0
    checksum_spec: str = ""
    checksum: int = 0
    creation_date: str = ""
    creation_time: str = ""
    patient_id: str = ""
    patient_ser: int = 0
    series_id: str = ""
    series_ser: int = 0
    slice_id: str = ""
    slice_ser: int = 0
    size_x: int = 0
    size_y: int = 0
    slice_z_pos: float = 0.0
    modality: str = ""
    window: int = 0
    level: int = 0
    pixel_offset: int = 0
    image_type: str = ""
    gantry_rtn: float = 0.0
    sad: float = 0.0
    sfd: float = 0.0
    coll_x1: float = 0.0
    coll_x2: float = 0.0
    coll_y1: float = 0.0
    coll_y2: float = 0.0
    coll_rtn: float = 0.0
    field_x: float = 0.0
    field_y: float = 0.0
    blade_x1: float = 0.0
    blade_x2: float = 0.0
    blade_y1: float = 0.0
    blade_y2: float = 0.0
    idu_pos_lng: float = 0.0
    idu_pos_lat: float = 0.0
    idu_pos_vrt: float = 0.0
    idu_pos_rtn: float = 0.0
    patient_support_angle: float = 0.0
    table_top_eccentric_angle: float = 0.0
    couch_vrt: float = 0.0
    couch_lng: float = 0.0
    couch_lat: float = 0.0
    idu_resolution_x: float = 0.0
    idu_resolution_y: float = 0.0
    image_resolution_x: float = 0.0
    image_resolution_y: float = 0.0
    energy: float = 0.0
    dose_rate: float = 0.0
    xray_kv: float = 0.0
    xray_ma: float = 0.0
    meterset_exposure: float = 0.0
    acq_adjustment: float = 0.0
    ct_projection_angle: float = 0.0
    cbct_positive_angle: float = 0.0
    ct_norm_chamber: float = 0.0
    gating_time_tag: float = 0.0
    gating_4d_info_x: float = 0.0
    gating_4d_info_y: float = 0.0
    gating_4d_info_z: float = 0.0
    gating_4d_info_time: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    unused_field: float = 0.0

def load_xim_angles_and_files(
    dirpath: DirPath,
    angle_type: Literal['kv-detector', 'kv-source', 'mv-source'] = 'kv-detector',
    closest_to: int | float | List[int | float] | None = None,
    n_angles: int | None = None,
    progress_callback=None,
    sort_by_angle: bool = True,
    start: int = 0,
    **kwargs,
    ) -> List[float]:
    closest_to = arg_to_list(closest_to, (int, float))

    # Get files.
    files = os.listdir(dirpath)
    files = [f for f in files if f.endswith(".xim")]
    files = list(sorted(files))
    if n_angles is None:
        n_angles = len(files) - start

    # Load filenames and angles.
    angles_files = {}
    for i in range(start, start + n_angles):
        filename = files[i]
        filepath = os.path.join(dirpath, filename)
        try:
            _, info = load_xim(filepath, load_pixel_data=False, **kwargs)
        except Exception as e:
            print(f"Error reading {filepath}, projection {i}: {e}")
            if progress_callback is not None:
                progress_callback(i - start + 1, n_angles)
            continue
        angles_files[info[f'{angle_type}-angle']] = filename
        if progress_callback is not None:
            progress_callback(i - start + 1, n_angles)

    # Filter by closest_to.
    if closest_to is not None:
        angles = np.array(list(angles_files.keys()))
        idxs_to_keep = []
        for c in closest_to:
            angle_diffs = np.abs(angles - c)
            idx = np.argmin(angle_diffs)
            idxs_to_keep.append(idx)
        angles_files = dict(t for i, t in enumerate(angles_files.items()) if i in idxs_to_keep)

    if sort_by_angle:
        angles_files = dict(sorted(angles_files.items()))

    return angles_files

def load_xim_arc(
    dirpath: DirPath,
    load_data: bool = True,
    raw: bool = False,
    show_progress: bool = True,
    **kwargs,
    ) -> Tuple[BatchImage2D | None, pd.DataFrame]:
    if raw:
        files = sorted(f for f in os.listdir(dirpath) if f.endswith('.xim'))
        images, infos = [], []
        for filename in tqdm(files, desc='Loading .xim files', disable=not show_progress):
            filepath = os.path.join(dirpath, filename)
            try:
                pixel_data, info = load_xim(filepath, **kwargs)
            except Exception as e:
                logger.warn(f"Skipping '{filename}': {e}")
                continue
            images.append(pixel_data)
            infos.append(info)
        if len(images) == 0:
            logger.warn(f"No valid .xim files found in '{dirpath}'.")
            return None, pd.DataFrame()
        images = np.stack(images, axis=0)
        infos = pd.DataFrame(infos).reset_index(drop=True)
    else:
        proc_dirpath = dirpath + '_proc'
        if load_data:
            images = load_numpy(os.path.join(proc_dirpath, 'images.npz'))
        else:
            images = None
        infos = load_csv(os.path.join(proc_dirpath, 'info.csv'), eval_cols=['det-offset', 'det-size', 'det-spacing'])
    return images, infos

def process_xim_session(
    dirpath: DirPath,
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    # Load raw .xim files.
    images, infos = load_xim_arc(dirpath, raw=True, show_progress=show_progress, **kwargs)
    if images is None:
        return

    # Save to _proc folder.
    proc_dirpath = dirpath + '_proc'
    os.makedirs(proc_dirpath, exist_ok=True)

    logger.info(f"Saving processed xim session to '{proc_dirpath}'.")
    save_numpy(images, os.path.join(proc_dirpath, 'images.npz'))

    # Drop complex columns not suitable for CSV.
    infos = infos.drop(columns=['properties'], errors='ignore')
    save_csv(infos, os.path.join(proc_dirpath, 'info.csv'))

def load_xim(
    filepath: str | Path,
    # This is just a convention, we could flip the image and make it look
    # like we're standing at the detector, looking toward the source, but
    # KIM convention is from source -> detector so let's go with that.
    flip_lr: bool = False,
    hist_eq: bool = False,
    invert_intensities: bool = False,
    load_pixel_data: bool = True,
    scale: Literal['IEC61217', 'VARIAN_IEC', 'VARIAN_STANDARD'] = 'VARIAN_STANDARD',
    ) -> Tuple[Image2D | None, pd.Series]:
    filepath = Path(filepath)
    info = XimInfo(file_name=str(filepath))

    with open(filepath, "rb") as fid:
        # ---- Header --------------------------------------------------------
        info.file_format_identifier = fid.read(8).decode("ascii", errors="replace")
        info.file_format_version = struct.unpack("<i", fid.read(4))[0]
        info.image_width = struct.unpack("<i", fid.read(4))[0]
        info.image_height = struct.unpack("<i", fid.read(4))[0]
        info.bits_per_pixel = struct.unpack("<i", fid.read(4))[0]
        info.bytes_per_pixel = struct.unpack("<i", fid.read(4))[0]
        info.compression_indicator = struct.unpack("<i", fid.read(4))[0]

        width = info.image_width
        height = info.image_height
        num_pixels = width * height

        # ---- Pixel data ----------------------------------------------------
        # The lookup-table-size field is always present (even for
        # uncompressed images) — see igtXimImageIO.cxx.
        pixel_data: np.ndarray | None = None
        lookup_table_size = struct.unpack("<i", fid.read(4))[0]

        if info.compression_indicator == 1:
            # Compressed
            # lookup_table_size is the number of *bytes* in the LUT.
            # Each byte packs 4 × 2-bit entries (LSB first).
            raw_lut = np.frombuffer(
                fid.read(lookup_table_size), dtype=np.uint8
            )

            compressed_pixel_buffer_size = struct.unpack("<i", fid.read(4))[0]

            if load_pixel_data:
                # First row + first pixel of second row are stored raw (int32)
                seed_count = width + 1
                n_compressed = num_pixels - seed_count

                # Work in int64 so prediction arithmetic cannot overflow.
                pixel_data = np.zeros(num_pixels, dtype=np.int64)
                pixel_data[:seed_count] = np.frombuffer(
                    fid.read(seed_count * 4), dtype=np.int32
                ).astype(np.int64)

                # Read the remaining compressed bytes in one go
                remaining_bytes = compressed_pixel_buffer_size - seed_count * 4
                raw_bytes = fid.read(remaining_bytes)

                # Decompress using variable-length differential encoding.
                # Unpack 2-bit LUT entries on the fly (same logic as C++).
                lut_idx = 0
                lut_off = 0
                raw_pos = 0
                for i in range(seed_count, num_pixels):
                    byte_val = raw_lut[lut_idx]
                    if lut_off == 0:
                        v = byte_val & 0x03
                        lut_off = 1
                    elif lut_off == 1:
                        v = (byte_val & 0x0C) >> 2
                        lut_off = 2
                    elif lut_off == 2:
                        v = (byte_val & 0x30) >> 4
                        lut_off = 3
                    else:
                        v = (byte_val & 0xC0) >> 6
                        lut_off = 0
                        lut_idx += 1

                    if v == 0:
                        diff = struct.unpack_from("<b", raw_bytes, raw_pos)[0]
                        raw_pos += 1
                    elif v == 1:
                        diff = struct.unpack_from("<h", raw_bytes, raw_pos)[0]
                        raw_pos += 2
                    else:
                        diff = struct.unpack_from("<i", raw_bytes, raw_pos)[0]
                        raw_pos += 4

                    pixel_data[i] = (
                        diff
                        + pixel_data[i - 1]
                        + pixel_data[i - width]
                        - pixel_data[i - width - 1]
                    )

                # Downcast from int64 working buffer to the file's native type
                if info.bytes_per_pixel == 2:
                    pixel_data = pixel_data.astype(np.int16)
                else:
                    pixel_data = pixel_data.astype(np.int32)

                pixel_data = pixel_data.reshape((width, height), order="F")
            else:
                fid.seek(compressed_pixel_buffer_size, 1)

            # Uncompressed buffer size field (present even for compressed data)
            _uncompressed_pixel_buffer_size = struct.unpack("<i", fid.read(4))[0]

        else:
            # Uncompressed
            uncompressed_pixel_buffer_size = struct.unpack("<i", fid.read(4))[0]

            if load_pixel_data:
                if info.bytes_per_pixel == 1:
                    count = uncompressed_pixel_buffer_size
                    pixel_data = np.frombuffer(fid.read(count), dtype=np.int8)
                elif info.bytes_per_pixel == 2:
                    count = uncompressed_pixel_buffer_size // 2
                    pixel_data = np.frombuffer(fid.read(count * 2), dtype=np.int16)
                else:
                    count = uncompressed_pixel_buffer_size // 4
                    pixel_data = np.frombuffer(fid.read(count * 4), dtype=np.int32)

                pixel_data = pixel_data.reshape((width, height), order="F")
            else:
                fid.seek(uncompressed_pixel_buffer_size, 1)

        # ---- Histogram -----------------------------------------------------
        number_of_bins = struct.unpack("<i", fid.read(4))[0]
        if number_of_bins > 0:
            info.histogram.number_of_bins = number_of_bins
            info.histogram.data = np.frombuffer(
                fid.read(number_of_bins * 4), dtype=np.int32
            ).copy()

        # ---- Properties ----------------------------------------------------
        number_of_properties = struct.unpack("<i", fid.read(4))[0]
        for _ in range(number_of_properties):
            prop_name_length = struct.unpack("<i", fid.read(4))[0]
            prop_name = fid.read(prop_name_length).decode("ascii", errors="replace")
            prop_type = struct.unpack("<i", fid.read(4))[0]

            if prop_type == 0:
                prop_value = struct.unpack("<i", fid.read(4))[0]
            elif prop_type == 1:
                prop_value = struct.unpack("<d", fid.read(8))[0]
            elif prop_type == 2:
                val_length = struct.unpack("<i", fid.read(4))[0]
                prop_value = fid.read(val_length).decode("ascii", errors="replace")
            elif prop_type == 4:
                val_length = struct.unpack("<i", fid.read(4))[0]
                prop_value = np.frombuffer(
                    fid.read(val_length), dtype=np.float64
                ).copy()
            elif prop_type == 5:
                val_length = struct.unpack("<i", fid.read(4))[0]
                prop_value = np.frombuffer(
                    fid.read(val_length), dtype=np.int32
                ).copy()
            else:
                print(
                    f"\n{prop_name}: Property type {prop_type} is not supported! "
                    "Aborting property decoding!"
                )
                break

            info.properties[prop_name] = prop_value

    # Add projection properties.
    metadata = {}
    if not 'GantryRtn' in info.properties:
        raise ValueError(f"No 'GantryRtn' param for .xim file: {filepath}")
    mv_source_angle = info.properties['GantryRtn']
    # Convert to IEC61217 if needed.
    if scale != 'IEC61217':
        mv_source_angle, _, _ = plc.core.scale.convert(
            input_scale=getattr(plc.core.scale.MachineScale, scale),
            output_scale=plc.core.scale.MachineScale.IEC61217,
            gantry=mv_source_angle,
            collimator=0,
            rotation=0,
        )
    metadata['mv-source-angle'] = mv_source_angle
    # Note! The .xim files KvSource/DetectorRtn properties may not be reliable, use GantryRtn.
    # We saw 'KVSourceRtn' always at 0 for Orange - Prostate - Pat01 - Fx01.
    # metadata['kv-detector-angle'] = float(np.round(info.properties['KVDetectorRtn'] % 360, decimals=3))
    # metadata['kv-source-angle'] = float(np.round(info.properties['KVSourceRtn'] % 360, decimals=3))
    # Calculate based on MV gantry.
    metadata['kv-source-angle'] = convert_angles(mv_source_angle, from_='mv-source', to='kv-source', machine='varian')
    metadata['kv-detector-angle'] = convert_angles(mv_source_angle, from_='mv-source', to='kv-detector', machine='varian')
    metadata['sid'] = float(np.round(info.properties['KVSourceVrt']) * 10)
    metadata['sdd'] = float(np.round((np.abs(info.properties['KVDetectorVrt']) + info.properties['KVSourceVrt']) * 10))
    metadata['det-offset'] = (float(np.round(np.abs(info.properties['KVDetectorLat'] * 10))), float(np.round(np.abs(info.properties['KVDetectorLng'] * 10))))
    metadata['det-size'] = (info.image_width, info.image_height)
    metadata['det-spacing'] = (info.properties['PixelWidth'] * 10, info.properties['PixelHeight'] * 10)
    metadata['properties'] = info.properties
    metadata['machine'] = 'varian'
    metadata['start-time'] = datetime.fromtimestamp(info.properties['StartTime']) + timedelta(microseconds=info.properties.get('StartTimeRemainder', 0) / 1000)
    metadata['stop-time'] = datetime.fromtimestamp(info.properties['StopTime']) + timedelta(microseconds=info.properties.get('StopTimeRemainder', 0) / 1000)
    metadata['start-time-s'] = info.properties['StartTime'] + info.properties.get('StartTimeRemainder', 0) / 1e9
    metadata['stop-time-s'] = info.properties['StopTime'] + info.properties.get('StopTimeRemainder', 0) / 1e9

    if flip_lr and pixel_data is not None:
        pixel_data = np.flip(pixel_data, axis=0)
    if pixel_data is not None:
        pixel_data = np.flip(pixel_data, axis=1)
    if invert_intensities and pixel_data is not None:
        pixel_data = np.max(pixel_data) - pixel_data

    if hist_eq and pixel_data is not None:
        pixel_data = hist_eq_fn(pixel_data)

    metadata = pd.Series(metadata)
    return pixel_data, metadata

def plot_xim_closest_to(
    dirpath: DirPath,
    angle: Literal['kv-detector', 'kv-source', 'mv-source'] = 'kv-source',
    closest_to: 'Number | List[Number]' = [0, 90, 180, 270, 360],
    **kwargs,
    ) -> None:
    angles_files = load_xim_angles_and_files(dirpath, angle, closest_to=closest_to, **kwargs)

    datas = []
    infos = []
    for f in angles_files.values():
        data, info = load_xim(os.path.join(dirpath, f))
        datas.append(data)
        infos.append(info)

    for d, i in zip(datas, infos):
        plot_projection(d, i)

def load_hnc(
    filepath: str | Path,
    flip_lr: bool = False,
    hist_eq: bool = False,
    invert_intensities: bool = False,
    load_pixel_data: bool = True,
    ) -> Tuple[np.ndarray | None, pd.Series]:
    filepath = Path(filepath)
    info = HncInfo(file_name=str(filepath))

    with open(filepath, "rb") as fid:
        # ---- Header (512 bytes total) --------------------------------------
        info.file_type           = fid.read(32).decode("ascii", errors="replace").rstrip("\x00")
        info.file_length         = struct.unpack("<I", fid.read(4))[0]
        info.actual_file_length  = filepath.stat().st_size
        info.checksum_spec       = fid.read(4).decode("ascii", errors="replace").rstrip("\x00")
        info.checksum            = struct.unpack("<I", fid.read(4))[0]
        info.creation_date       = fid.read(8).decode("ascii", errors="replace").rstrip("\x00")
        info.creation_time       = fid.read(8).decode("ascii", errors="replace").rstrip("\x00")
        info.patient_id          = fid.read(16).decode("ascii", errors="replace").rstrip("\x00")
        info.patient_ser         = struct.unpack("<I", fid.read(4))[0]
        info.series_id           = fid.read(16).decode("ascii", errors="replace").rstrip("\x00")
        info.series_ser          = struct.unpack("<I", fid.read(4))[0]
        info.slice_id            = fid.read(16).decode("ascii", errors="replace").rstrip("\x00")
        info.slice_ser           = struct.unpack("<I", fid.read(4))[0]
        info.size_x              = struct.unpack("<I", fid.read(4))[0]
        info.size_y              = struct.unpack("<I", fid.read(4))[0]
        info.slice_z_pos         = struct.unpack("<d", fid.read(8))[0]
        info.modality            = fid.read(16).decode("ascii", errors="replace").rstrip("\x00")
        info.window              = struct.unpack("<I", fid.read(4))[0]
        info.level               = struct.unpack("<I", fid.read(4))[0]
        info.pixel_offset        = struct.unpack("<I", fid.read(4))[0]
        info.image_type          = fid.read(4).decode("ascii", errors="replace").rstrip("\x00")
        info.gantry_rtn          = struct.unpack("<d", fid.read(8))[0]
        info.sad                 = struct.unpack("<d", fid.read(8))[0]
        info.sfd                 = struct.unpack("<d", fid.read(8))[0]
        info.coll_x1             = struct.unpack("<d", fid.read(8))[0]
        info.coll_x2             = struct.unpack("<d", fid.read(8))[0]
        info.coll_y1             = struct.unpack("<d", fid.read(8))[0]
        info.coll_y2             = struct.unpack("<d", fid.read(8))[0]
        info.coll_rtn            = struct.unpack("<d", fid.read(8))[0]
        info.field_x             = struct.unpack("<d", fid.read(8))[0]
        info.field_y             = struct.unpack("<d", fid.read(8))[0]
        info.blade_x1            = struct.unpack("<d", fid.read(8))[0]
        info.blade_x2            = struct.unpack("<d", fid.read(8))[0]
        info.blade_y1            = struct.unpack("<d", fid.read(8))[0]
        info.blade_y2            = struct.unpack("<d", fid.read(8))[0]
        info.idu_pos_lng         = struct.unpack("<d", fid.read(8))[0]
        info.idu_pos_lat         = struct.unpack("<d", fid.read(8))[0]
        info.idu_pos_vrt         = struct.unpack("<d", fid.read(8))[0]
        info.idu_pos_rtn         = struct.unpack("<d", fid.read(8))[0]
        info.patient_support_angle       = struct.unpack("<d", fid.read(8))[0]
        info.table_top_eccentric_angle   = struct.unpack("<d", fid.read(8))[0]
        info.couch_vrt           = struct.unpack("<d", fid.read(8))[0]
        info.couch_lng           = struct.unpack("<d", fid.read(8))[0]
        info.couch_lat           = struct.unpack("<d", fid.read(8))[0]
        info.idu_resolution_x    = struct.unpack("<d", fid.read(8))[0]
        info.idu_resolution_y    = struct.unpack("<d", fid.read(8))[0]
        info.image_resolution_x  = struct.unpack("<d", fid.read(8))[0]
        info.image_resolution_y  = struct.unpack("<d", fid.read(8))[0]
        info.energy              = struct.unpack("<d", fid.read(8))[0]
        info.dose_rate           = struct.unpack("<d", fid.read(8))[0]
        info.xray_kv             = struct.unpack("<d", fid.read(8))[0]
        info.xray_ma             = struct.unpack("<d", fid.read(8))[0]
        info.meterset_exposure   = struct.unpack("<d", fid.read(8))[0]
        info.acq_adjustment      = struct.unpack("<d", fid.read(8))[0]
        info.ct_projection_angle = struct.unpack("<d", fid.read(8))[0]
        info.cbct_positive_angle = (info.ct_projection_angle + 270.0) % 360.0
        info.ct_norm_chamber     = struct.unpack("<d", fid.read(8))[0]
        info.gating_time_tag     = struct.unpack("<d", fid.read(8))[0]
        info.gating_4d_info_x    = struct.unpack("<d", fid.read(8))[0]
        info.gating_4d_info_y    = struct.unpack("<d", fid.read(8))[0]
        info.gating_4d_info_z    = struct.unpack("<d", fid.read(8))[0]
        info.gating_4d_info_time = struct.unpack("<d", fid.read(8))[0]
        info.offset_x            = struct.unpack("<d", fid.read(8))[0]
        info.offset_y            = struct.unpack("<d", fid.read(8))[0]
        info.unused_field        = struct.unpack("<d", fid.read(8))[0]

        # ---- Pixel data ----------------------------------------------------
        pixel_data: np.ndarray | None = None
        if load_pixel_data:
            n_pixels = info.size_x * info.size_y
            remaining = info.actual_file_length - fid.tell()
            if remaining % n_pixels != 0:
                raise ValueError(
                    f"Incompatible HNC file format: {remaining} remaining bytes "
                    f"not divisible by {n_pixels} pixels in '{filepath}'."
                )
            bytes_per_pixel = remaining // n_pixels
            if bytes_per_pixel == 1:
                pixel_data = np.frombuffer(fid.read(n_pixels), dtype=np.uint8)
            elif bytes_per_pixel == 2:
                pixel_data = np.frombuffer(fid.read(n_pixels * 2), dtype=np.uint16)
            elif bytes_per_pixel == 4:
                pixel_data = np.frombuffer(fid.read(n_pixels * 4), dtype=np.uint32)
            else:
                raise ValueError(
                    f"Incompatible HNC file format: {bytes_per_pixel} bytes/pixel "
                    f"in '{filepath}'."
                )
            pixel_data = pixel_data.reshape((info.size_x, info.size_y), order="F")

    metadata = pd.Series({
        'mv-source-angle':      info.gantry_rtn,
        'ct-projection-angle':  info.ct_projection_angle,
        'cbct-positive-angle':  info.cbct_positive_angle,
        'sid':                  info.sad,
        'sdd':                  info.sfd,
        'det-size':             (info.size_x, info.size_y),
        'det-spacing':          (info.idu_resolution_x, info.idu_resolution_y),
        'info':                 info,
        'machine':              'varian',
    })

    if flip_lr and pixel_data is not None:
        pixel_data = np.flip(pixel_data, axis=0)
    if invert_intensities and pixel_data is not None:
        pixel_data = np.max(pixel_data) - pixel_data
    if hist_eq and pixel_data is not None:
        pixel_data = hist_eq_fn(pixel_data)

    return pixel_data, metadata
