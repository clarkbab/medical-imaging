from dataclasses import dataclass, field
import numpy as np
import os
from pathlib import Path
import struct
from typing import Any, Tuple

from mymi.typing import *

from .args import arg_to_list
from .projections import reverse_angles

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

def load_xim_angles_and_files(
    dirpath: DirPath,
    angle_type: Literal['kv-detector', 'kv-source', 'mv-source'],
    closest_to: int | float | List[int | float] | None = None,
    n_angles: int | None = None,
    sort_by_angle: bool = True,
    start: int = 0,
    **kwargs,
    ) -> List[float]:
    closest_to = arg_to_list(closest_to, (int, float, None))

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
            continue
        angles_files[info[f'{angle_type}-angle']] = filename

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

def load_xim(
    filepath: str | Path,
    flip_lr: bool = True,
    invert_intensities: bool = True,
    load_pixel_data: bool = True,
    reverse_gantry_angle: bool = True,
    ) -> Tuple[np.ndarray | None, XimInfo]:
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

    # Add projection properies.
    metadata = {}
    # I think that .xim files encode MV gantry using a CCW+ convention. This may not
    # be true for all machines, so have the option to flip. See the discussion at:
    # "PRJ-LEARN:\ProjectData\Brett\XIM angles confusion.docx".
    mv_source_angle = float(np.round(info.properties['GantryRtn'] % 360, decimals=3))
    if reverse_gantry_angle:
        mv_source_angle = reverse_angles(mv_source_angle)
    metadata['mv-source-angle'] = mv_source_angle
    # Note! The .xim files KvSource/DetectorRtn properties may not be reliable, use GantryRtn.
    # We saw 'KVSourceRtn' always at 0 for Orange - Prostate - Pat01 - Fx01.
    # metadata['kv-detector-angle'] = float(np.round(info.properties['KVDetectorRtn'] % 360, decimals=3))
    # metadata['kv-source-angle'] = float(np.round(info.properties['KVSourceRtn'] % 360, decimals=3))
    # Calculate based on MV gantry.
    metadata['kv-source-angle'] = float(np.round((mv_source_angle - 90) % 360, decimals=3))
    metadata['kv-detector-angle'] = float(np.round((mv_source_angle + 90) % 360, decimals=3))
    metadata['sid'] = float(np.round(info.properties['KVSourceVrt']) * 10)
    metadata['sdd'] = float(np.round((np.abs(info.properties['KVDetectorVrt']) + info.properties['KVSourceVrt']) * 10))
    metadata['det-offset'] = (float(np.round(np.abs(info.properties['KVDetectorLat'] * 10))), float(np.round(np.abs(info.properties['KVDetectorLng'] * 10))))
    metadata['det-size'] = (info.image_width, info.image_height)
    metadata['det-spacing'] = (info.properties['PixelWidth'] * 10, info.properties['PixelHeight'] * 10)
    metadata['properties'] = info.properties

    if flip_lr and pixel_data is not None:
        pixel_data = np.flip(pixel_data, axis=0)
    if invert_intensities and pixel_data is not None:
        pixel_data = np.max(pixel_data) - pixel_data

    return pixel_data, metadata
