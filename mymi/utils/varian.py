from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import struct
from typing import Any, Tuple

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

def read_xim(
    filepath: str | Path,
    read_pixel_data: bool = True,
    ) -> Tuple[XimInfo, np.ndarray | None]:
    """
    Read a Varian XIM image file.

    Parameters
    ----------
    filepath : str or Path
        Full path to the .xim file.
    read_pixel_data : bool, optional
        Whether to decode the pixel data (default True). Set to False to read
        only the header/properties.

    Returns
    -------
    info : XimInfo
        Header information and properties.
    image : np.ndarray or None
        2-D image array (width × height), or None if read_pixel_data is False.
    """
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

            if read_pixel_data:
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

            if read_pixel_data:
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

    return info, pixel_data
