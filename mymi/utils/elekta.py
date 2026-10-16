from dicomset.typing import *
from dicomset.utils import hist_eq as hist_eq_fn, save_csv, save_numpy
from dataclasses import dataclass, field
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pylinac as plc
import struct
from tqdm.auto import tqdm
from typing import Dict, List, Literal, Tuple
from xml.etree import ElementTree

from mymi.logging import logger

from .args import arg_to_list
from .projections import convert_angles

ELEKTA_DET_SIZE_X = 409.6  # mm
ELEKTA_DET_SIZE_Y = 409.6  # mm
ELEKTA_XVI_SID = 1000.0    # mm, source-to-isocentre distance
ELEKTA_XVI_SDD = 1536.0    # mm, source-to-detector distance
HIS_HEAD_LENGTH = 68

# Correction bitmask flags (byte offset 22-23).
HIS_CORRECTION_DARK = 1 << 0
HIS_CORRECTION_GAIN = 1 << 1
HIS_CORRECTION_BAD_PIXEL = 1 << 2

# TypeOfNumbers values.
HIS_TYPE_UINT8 = 2
HIS_TYPE_UINT16 = 4
HIS_TYPE_UINT32 = 8
HIS_TYPE_UINT64 = 16


@dataclass
class HisInfo:
    file_type: int = 0            # Magic number (0x7000).
    header_size: int = 0          # Total header bytes (fixed 68 + image header).
    header_version: int = 0       # Format version (typically 100).
    file_size_blocks: int = 0     # File size in 1024-byte blocks.
    image_header_size: int = 0    # Extra header bytes beyond the fixed 68.
    ulx: int = 0                  # Upper-left X of ROI on detector.
    uly: int = 0                  # Upper-left Y of ROI on detector.
    brx: int = 0                  # Bottom-right X of ROI on detector.
    bry: int = 0                  # Bottom-right Y of ROI on detector.
    size_x: int = 0               # Image width  (bry - uly + 1).
    size_y: int = 0               # Image height (brx - ulx + 1).
    n_frames: int = 0             # Number of frames.
    correction: int = 0           # Bitmask of applied corrections.
    integration_time: float = 0.0 # Exposure/integration time (ms).
    data_type: int = 0            # TypeOfNumbers (2=u8, 4=u16, 8=u32, 16=u64).
    pixel_spacing_x: float = 0.0
    pixel_spacing_y: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0


@dataclass
class HisFrameInfo:
    """Per-frame metadata from ``_Frames.xml``."""
    frame: int = 0                  # 1-based frame number (matches .his filename prefix).
    mv_source_angle: float = 0.0    # Gantry angle in degrees.
    u_centre: float = 0.0        # Detector U (lateral) offset in mm — large value = half-fan.
    v_centre: float = 0.0        # Detector V (longitudinal) offset in mm.
    delta_ms: float = 0.0        # Time offset from first frame in ms.
    exposed: bool = True
    mv_on: bool = False
    inactive: bool = False
    has_pixel_factor: bool = False
    pixel_factor: float = 0.0
    filename: str = ""            # Corresponding .his filename (populated by load_his_angles_and_files).

# In PA x-ray imaging, the source is posterior and the detector is anterior. The images
# are viewed from detector to source, so patient right appears on the left and vice versa.
# Ref: https://www.glowm.com/atlas-page/atlasid/chestXray.html
# So for our kV x-ray images, if we have kv source to the patient's right (270 degrees)
# then the kv detector will be to the patient's left (90 degrees). Then it will look like
# we're standing at the patient's left. To make this work for the sample treatment images,
# I needed to flip_lr the image.
def load_his(
    filepath: str | Path,
    # This is just a convention, we could flip the image and make it look
    # like we're standing at the detector, looking toward the source, but
    # KIM convention is from source -> detector so let's go with that.
    flip_lr: bool = False,
    frame_info: HisFrameInfo | None = None,
    hist_eq: bool = False,
    load_pixel_data: bool = True,
    scale: Literal['IEC61217', 'ELEKTA_IEC'] = 'IEC61217',
    sid: float = ELEKTA_XVI_SID,
    sdd: float = ELEKTA_XVI_SDD,
    ) -> Tuple[Image2D | None, pd.Series]:
    filepath = Path(filepath)

    with open(filepath, "rb") as fid:
        # ---- Fixed 68-byte header ------------------------------------------
        raw = fid.read(HIS_HEAD_LENGTH)
        if len(raw) < HIS_HEAD_LENGTH:
            raise ValueError(f"File '{filepath}' is too short to be a HIS file")

        # Unpack header using little-endian unsigned types at known offsets.
        #   0- 1  uint16  FileType          (magic 0x7000)
        #   2- 3  uint16  HeaderSize        (fixed part, always 68)
        #   4- 5  uint16  HeaderVersion     (100)
        #   6- 9  uint32  FileSize          (in 1024-byte blocks)
        #  10-11  uint16  ImageHeaderSize   (extra bytes beyond 68)
        #  12-13  uint16  ULX
        #  14-15  uint16  ULY
        #  16-17  uint16  BRX
        #  18-19  uint16  BRY
        #  20-21  uint16  NrOfFrames
        #  22-23  uint16  Correction        (bitmask)
        #  24-31  float64 IntegrationTime
        #  32-33  uint16  TypeOfNumbers
        #  34-67  reserved
        (file_type, fixed_hdr_size, header_version,
         file_size_blocks,
         image_header_size,
         ulx, uly, brx, bry,
         n_frames, correction,
         integration_time,
         data_type) = struct.unpack_from('<HHH I H HHHHH H d H', raw, 0)

        # Validate magic bytes.
        if file_type != 0x7000:
            raise ValueError(
                f"File '{filepath}' is not in Heimann HIS format "
                f"(expected magic 0x7000, got 0x{file_type:04X})"
            )

        info = HisInfo()
        info.file_type = file_type
        info.header_version = header_version
        info.file_size_blocks = file_size_blocks
        info.image_header_size = image_header_size
        info.header_size = image_header_size + HIS_HEAD_LENGTH
        info.ulx = ulx
        info.uly = uly
        info.brx = brx
        info.bry = bry
        info.n_frames = n_frames
        info.correction = correction
        info.integration_time = integration_time
        info.data_type = data_type

        info.size_x = bry - uly + 1
        info.size_y = brx - ulx + 1
        info.pixel_spacing_x = ELEKTA_DET_SIZE_X / info.size_x
        info.pixel_spacing_y = ELEKTA_DET_SIZE_Y / info.size_y
        info.origin_x = -0.5 * (info.size_x - 1) * info.pixel_spacing_x
        info.origin_y = -0.5 * (info.size_y - 1) * info.pixel_spacing_y

        # ---- Pixel data ----------------------------------------------------
        pixel_data: np.ndarray | None = None
        if load_pixel_data:
            n_pixels = info.n_frames * info.size_x * info.size_y

            # Choose dtype based on TypeOfNumbers.
            dtype_map = {
                HIS_TYPE_UINT8: np.uint8,
                HIS_TYPE_UINT16: np.uint16,
                HIS_TYPE_UINT32: np.uint32,
                HIS_TYPE_UINT64: np.uint64,
            }
            dtype = dtype_map.get(info.data_type, np.uint16)
            bytes_per_pixel = dtype().itemsize

            fid.seek(info.header_size)
            pixel_data = np.frombuffer(
                fid.read(n_pixels * bytes_per_pixel), dtype=dtype
            ).copy()

            if info.n_frames == 1:
                pixel_data = pixel_data.reshape((info.size_x, info.size_y), order="F")
            else:
                pixel_data = pixel_data.reshape(
                    (info.n_frames, info.size_y, info.size_x), order="F"
                )

    # ---- Build metadata dict (mirrors load_xim output) ---------------------
    metadata: Dict = {}
    metadata['filename'] = os.path.basename(filepath)
    metadata['det-size'] = (info.size_x, info.size_y)
    metadata['det-spacing'] = (info.pixel_spacing_x, info.pixel_spacing_y)
    metadata['integration-time'] = info.integration_time
    metadata['correction'] = info.correction
    metadata['sid'] = sid
    metadata['sdd'] = sdd
    metadata['his-info'] = info
    metadata['machine'] = 'elekta'

    # Load per-frame geometry from _Frames.xml (unless already provided).
    if frame_info is None:
        xml_path = filepath.parent / "_Frames.xml"
        if not xml_path.exists():
            logger.warning(f"No '_Frames.xml' found at '{xml_path}' — angle metadata will be missing.")
        else:
            all_frames = load_his_frames_xml(xml_path)
            frame = _frame_from_filename(filepath.name)
            if frame is None:
                logger.warning(f"Could not extract frame number from filename '{filepath.name}' — angle metadata will be missing.")
            else:
                for f in all_frames:
                    if f.frame == frame:
                        frame_info = f
                        break
                if frame_info is None:
                    logger.warning(f"No frame with frame={frame} found in '{xml_path}' — angle metadata will be missing.")

    if frame_info is not None:
        mv_source_angle = float(np.round(frame_info.mv_source_angle % 360, decimals=3))
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
        metadata['kv-source-angle'] = convert_angles(mv_source_angle, from_='mv-source', to='kv-source', machine='elekta')
        metadata['kv-detector-angle'] = convert_angles(mv_source_angle, from_='mv-source', to='kv-detector', machine='elekta')
        metadata['det-offset'] = (frame_info.u_centre, frame_info.v_centre)
        metadata['start-time-s'] = frame_info.delta_ms / 1000.0
        metadata['frame-info'] = frame_info

    if pixel_data is not None:
        if flip_lr:
            pixel_data = np.flip(pixel_data, axis=0)
        pixel_data = np.flip(pixel_data, axis=1)

    if hist_eq and pixel_data is not None:
        pixel_data = hist_eq_fn(pixel_data)

    metadata = pd.Series(metadata)
    return pixel_data, metadata

def load_his_frames_xml(
    filepath: str | Path,
    ) -> List[HisFrameInfo]:
    logger.info(f"Loading frame metadata from '{filepath}'...")
    filepath = Path(filepath)
    tree = ElementTree.parse(filepath)
    root = tree.getroot()

    frames: List[HisFrameInfo] = []
    for frame_el in root.iter("Frame"):
        f = HisFrameInfo()
        f.frame = int(_xml_text(frame_el, "Seq", "0"))
        f.mv_source_angle = float(_xml_text(frame_el, "GantryAngle", "0"))
        f.u_centre = float(_xml_text(frame_el, "UCentre", "0"))
        f.v_centre = float(_xml_text(frame_el, "VCentre", "0"))
        f.delta_ms = float(_xml_text(frame_el, "DeltaMs", "0"))
        f.exposed = _xml_text(frame_el, "Exposed", "True") == "True"
        f.mv_on = _xml_text(frame_el, "MVOn", "False") == "True"
        f.inactive = _xml_text(frame_el, "Inactive", "False") == "True"
        f.has_pixel_factor = _xml_text(frame_el, "HasPixelFactor", "False") == "True"
        f.pixel_factor = float(_xml_text(frame_el, "PixelFactor", "0"))
        frames.append(f)

    frames.sort(key=lambda x: x.frame)
    return frames

def load_his_angles_and_files(
    dirpath: str | Path,
    angle_type: Literal['kv-detector', 'kv-source', 'mv-source'] = 'kv-source',
    closest_to: int | float | List[int | float] | None = None,
    n_angles: int | None = None,
    scale: Literal['IEC61217', 'ELEKTA_IEC'] = 'IEC61217',
    sort_by_angle: bool = True,
    start: int = 0,
    ) -> Tuple[Dict[float, str], Dict[str, 'HisFrameInfo']]:
    dirpath = Path(dirpath)
    closest_to = arg_to_list(closest_to, (int, float))

    # Parse _Frames.xml once.
    xml_path = dirpath / "_Frames.xml"
    if not xml_path.exists():
        raise FileNotFoundError(
            f"No _Frames.xml found in '{dirpath}'. Cannot determine gantry angles."
        )
    frame_infos = load_his_frames_xml(xml_path)

    # Build frame → filename lookup from .his files on disk.
    his_files = list(f for f in os.listdir(dirpath) if f.endswith(".his"))
    frame_to_file: Dict[int, str] = {}
    for fname in his_files:
        print(fname)
        frame = _frame_from_filename(fname)
        print(frame)
        frame_to_file[frame] = fname

    # Build angle → filename dict and filename → HisFrameInfo dict.
    angles_files: Dict[float, str] = {}
    fi_by_name: Dict[str, HisFrameInfo] = {}
    for f in frame_infos:
        if f.frame not in frame_to_file:
            continue
        f.filename = frame_to_file[f.frame]
        mv_angle = float(np.round(f.mv_source_angle % 360, decimals=3))
        if scale != 'IEC61217':
            mv_angle, _, _ = plc.core.scale.convert(
                input_scale=getattr(plc.core.scale.MachineScale, scale),
                output_scale=plc.core.scale.MachineScale.IEC61217,
                gantry=mv_angle,
                collimator=0,
                rotation=0,
            )
        angle = convert_angles(mv_angle, from_='mv-source', to=angle_type, machine='elekta')
        angles_files[angle] = f.filename
        fi_by_name[f.filename] = f

    # Apply start / n_angles slicing.
    items = list(angles_files.items())
    if n_angles is None:
        n_angles = len(items) - start
    items = items[start:start + n_angles]
    angles_files = dict(items)

    # Filter by closest_to.
    if closest_to is not None:
        angles = np.array(list(angles_files.keys()))
        idxs_to_keep = []
        for c in closest_to:
            angle_diffs = np.abs(angles - c)
            idx = int(np.argmin(angle_diffs))
            idxs_to_keep.append(idx)
        angles_files = dict(
            t for i, t in enumerate(angles_files.items()) if i in idxs_to_keep
        )

    if sort_by_angle:
        angles_files = dict(sorted(angles_files.items()))

    # Restrict frame_info_by_file to only the selected files.
    frame_info_by_file = {fname: fi_by_name[fname] for fname in angles_files.values() if fname in fi_by_name}

    return angles_files, frame_info_by_file

def load_his_session(
    session_path: str | Path,
    load_data: bool = True,
    n_frames: int | None = None,
    raw: bool = False,
    show_progress: bool = True,
    ) -> Tuple[BatchImage2D | None, pd.DataFrame]:
    from dicomset.utils import load_csv, load_numpy
    session_path = Path(session_path)
    proc_path = Path(str(session_path) + '_proc')

    if raw:
        # No need for the angle-based filtering/sorting of 'load_his_angles_and_files' here —
        # frames are re-sorted by 'start-time-s' below anyway, and every .his file in the
        # folder is loaded regardless of its gantry angle.
        xml_path = session_path / "_Frames.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"No _Frames.xml found in '{session_path}'. Cannot determine frame metadata.")
        frame_info_by_num = {f.frame: f for f in load_his_frames_xml(xml_path)}
        files = sorted(
            (f for f in os.listdir(session_path) if f.endswith(".his")),
            key=lambda f: _frame_from_filename(f),
        )
        if n_frames is not None:
            files = files[:n_frames]
        images, rows = [], []
        for fname in tqdm(files, desc='Loading HIS frames', disable=not show_progress):
            frame = _frame_from_filename(fname)
            fi = frame_info_by_num.get(frame)
            img, info = load_his(session_path / fname, frame_info=fi, load_pixel_data=load_data)
            images.append(img)
            rows.append(info)
        images = np.stack(images, axis=0) if load_data else None
        infos = pd.DataFrame(rows)
        # Default of filename sorting doesn't work for HIS.
        infos = infos.sort_values(by='start-time-s').reset_index(drop=True)
    else:
        if load_data:
            images = load_numpy(str(proc_path / 'images.npz')).astype(np.float32)
        else:
            images = None
        infos = load_csv(str(proc_path / 'info.csv'), eval_cols=['det-size', 'det-spacing', 'det-offset'])
        if n_frames is not None:
            if images is not None:
                images = images[:n_frames]
            infos = infos.iloc[:n_frames]

    return images, infos

def create_his_patient_videos(
    pat_path: str | Path,
    fraction: 'int | List[int] | Literal["all"]' = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    from glob import glob
    pat_path = str(pat_path)
    if fraction == 'all':
        fractions = sorted(
            int(os.path.basename(p).upper().replace('FX', ''))
            for p in glob(os.path.join(pat_path, 'FX*'))
            if os.path.isdir(p)
        )
    else:
        fractions = arg_to_list(fraction, int)
    for f in tqdm(fractions, desc='Fraction videos', disable=not show_progress):
        frac_path = os.path.join(pat_path, f'FX{f}')
        create_his_fraction_videos(frac_path, show_progress=show_progress, **kwargs)

def create_his_fraction_videos(
    frac_path: str | Path,
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    from glob import glob
    pattern = os.path.join(str(frac_path), 'CBCT', 'CBCT*', 'CBCT Projections', 'XVI', '*')
    session_paths = sorted(p for p in glob(pattern) if os.path.isdir(p) and not p.endswith('_proc'))
    for session_path in tqdm(session_paths, desc='Session videos', disable=not show_progress):
        create_his_session_video(session_path, show_progress=show_progress, **kwargs)

def create_his_session_video(
    session_path: str | Path,
    hist_eq: bool = True,
    show_progress: bool = True,
    ) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dicomset.utils import load_numpy, hist_eq as hist_eq_fn
    from mymi.utils.plotting import play_video
    import PIL.Image
    import io as _io

    proc_path = str(session_path) + '_proc'
    images_path = os.path.join(proc_path, 'images.npz')
    if not os.path.exists(images_path):
        logger.warning(f"No images.npz at '{images_path}' — run process_his_session first.")
        return

    images = load_numpy(images_path)   # (n_frames, H, W)
    n_frames = images.shape[0]

    def _plot_frame(frame_data: np.ndarray) -> PIL.Image.Image:
        d = frame_data.astype(np.float64)
        if hist_eq:
            d = hist_eq_fn(d)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(d.T, cmap='gray', origin='lower', aspect='auto')
        ax.axis('off')
        fig.tight_layout(pad=0)
        buf = _io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return PIL.Image.open(buf).copy()

    plot_args = [(images[i],) for i in range(n_frames)]
    savepath = os.path.join(proc_path, 'video.mp4')
    play_video(_plot_frame, plot_args, overwrite=True, savepath=savepath,
               show=False, show_progress=show_progress)
    logger.info(f"Saved video to '{savepath}'.")

def create_his_patient_shrouds(
    pat_path: str | Path,
    fraction: 'int | List[int] | Literal["all"]' = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    from glob import glob
    pat_path = str(pat_path)
    if fraction == 'all':
        fractions = sorted(
            int(os.path.basename(p).upper().replace('FX', ''))
            for p in glob(os.path.join(pat_path, 'FX*'))
            if os.path.isdir(p)
        )
    else:
        fractions = arg_to_list(fraction, int)
    for f in tqdm(fractions, desc='Fraction shrouds', disable=not show_progress):
        frac_path = os.path.join(pat_path, f'FX{f}')
        create_his_fraction_shrouds(frac_path, show_progress=show_progress, **kwargs)

def create_his_fraction_shrouds(
    frac_path: str | Path,
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    from glob import glob
    pattern = os.path.join(str(frac_path), 'CBCT', 'CBCT*', 'CBCT Projections', 'XVI', '*')
    session_paths = sorted(p for p in glob(pattern) if os.path.isdir(p) and not p.endswith('_proc'))
    for session_path in tqdm(session_paths, desc='Session shrouds', disable=not show_progress):
        create_his_session_shroud(session_path, show_progress=show_progress, **kwargs)

def create_his_session_shroud(
    session_path: str | Path,
    show_progress: bool = True,
    ) -> None:
    from dicomset.utils import load_numpy
    proc_path = str(session_path) + '_proc'
    images_path = os.path.join(proc_path, 'images.npz')
    if not os.path.exists(images_path):
        logger.warning(f"No images.npz found at '{images_path}' — run process_his_session first.")
        return
    images = load_numpy(images_path)   # (n_frames, H, W)
    for i in tqdm(range(images.shape[0]), desc='Normalising frames', disable=not show_progress):
        images[i] = hist_eq_fn(images[i])
    shroud = images.mean(axis=1)       # (n_frames, W)
    save_numpy(shroud, os.path.join(proc_path, 'shroud.npz'))
    logger.info(f"Saved shroud to '{proc_path}'.")

def process_his_fraction(
    fracpath: str | Path,
    show_progress: bool = True,
    ) -> None:
    """Process all CBCT HIS sessions found under a fraction directory.

    Expected layout: <fracpath>/CBCT/CBCT*/CBCT Projections/XVI/*/
    """
    from glob import glob
    pattern = os.path.join(str(fracpath), 'CBCT', 'CBCT*', 'CBCT Projections', 'XVI', '*')
    session_paths = sorted(p for p in glob(pattern) if os.path.isdir(p) and not p.endswith('_proc'))
    if not session_paths:
        logger.warning(f"No HIS session directories found under '{fracpath}'.")
        return
    for session_path in tqdm(session_paths, desc='CBCT sessions', disable=not show_progress):
        process_his_session(session_path, show_progress=show_progress)

def process_his_session(
    dirpath: str | Path,
    show_progress: bool = True,
    ) -> None:
    dirpath = Path(dirpath)
    images, infos = load_his_session(dirpath, raw=True, show_progress=show_progress)

    proc_dirpath = str(dirpath) + '_proc'
    os.makedirs(proc_dirpath, exist_ok=True)

    logger.info(f"Saving processed HIS session to '{proc_dirpath}'.")
    save_numpy(images, os.path.join(proc_dirpath, 'images.npz'))
    infos = infos.drop(columns=['his-info', 'frame-info'], errors='ignore')
    save_csv(infos, os.path.join(proc_dirpath, 'info.csv'))

def plot_his_closest_to(
    dirpath: str | Path,
    angle: Literal['kv-detector', 'kv-source', 'mv-source'] = 'kv-source',
    closest_to: 'Number | List[Number]' = [0, 90, 180, 270, 360],
    **kwargs,
    ) -> None:
    from .projections import plot_projection
    angles_files, frame_info_by_file = load_his_angles_and_files(dirpath, angle, closest_to=closest_to, **kwargs)

    datas = []
    infos = []
    for f in angles_files.values():
        fi = frame_info_by_file.get(f)
        data, info = load_his(Path(dirpath) / f, frame_info=fi)
        datas.append(data)
        infos.append(info)

    for d, i in zip(datas, infos):
        plot_projection(d, i)

def _xml_text(parent, tag: str, default: str = "") -> str:
    """Get text content of a child element, or *default* if missing."""
    el = parent.find(tag)
    if el is not None and el.text is not None:
        return el.text.strip()
    return default

def _frame_from_filename(filename: str) -> int | None:
    prefix = filename.split(".")[0]
    try:
        return int(prefix)
    except ValueError:
        return None
