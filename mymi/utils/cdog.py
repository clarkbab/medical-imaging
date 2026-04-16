import datetime
from importlib.metadata import metadata
from dicomset.typing import *
from dicomset.utils import bubble_args, to_list
import imageio
import io
from IPython.display import Image as IPythonImage, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import PIL
import seaborn as sns
from skimage import exposure
import tempfile
from tqdm import tqdm
from typing import *

from mymi.typing import *

from .angles import convert_angle
from .args import arg_to_list
from .io import resolve_filepath
from .pandas import append_row

KV_FOLDERS = ['KIM-KV', 'kV']

def get_base_pixel(
    metadata: np.ndarray,
    base_flag: int = int('0xCAFE', 16),
    ) -> Tuple[int, int]:
    
    # Get the base pixel.
    for i, row in enumerate(metadata):
        for j, pixel in enumerate(row):
            if pixel == base_flag:
                return (i, j)

    raise ValueError(f"No base_pixel found with value: {base_flag}")

def __load_tiff_metadata(
    metadata_img: np.ndarray,
    cdog_version: Literal['v2.7', 'v3.0', 'v4.0'],
    base_flag: int = int('0xCAFE', 16),
    ) -> Dict[str, Any]:
    base_pixel = get_base_pixel(metadata_img, base_flag=base_flag)
    prop_df = load_prop_df(cdog_version)
    props = {}

    # Add datetimes.
    prop_types = {
        'StartTime': 'datetime',
        'StopTime': 'datetime',
    }

    # Add all doubles.
    doubles = prop_df[prop_df['datatype'] == 1]
    doubles = doubles['prop'].tolist()
    for d in doubles:
        prop_types[d] = 'double'
    
    for p, t in prop_types.items():
        if t == 'datetime':
            v = load_timestamp(metadata_img, base_flag=base_flag, base_pixel=base_pixel, prop=p, prop_df=prop_df)
        elif t == 'double':
            v = load_double(metadata_img, base_flag=base_flag, base_pixel=base_pixel, prop=p, prop_df=prop_df)
        else:
            raise ValueError(f"Unrecognised type '{t}'.")
            
        props[p] = v

    # Extract parameters required for DRR creation.
    props['det-offset'] = (float(np.round(props['KVDetectorLat'] * 10)), float(np.round(props['KVDetectorLng'] * 10)))
    props['det-spacing'] = (float(props['PixelWidth'] * 10), float(props['PixelHeight'] * 10))
    props['sid'] = float(np.round(props['KVSourceVrt'] * 10))
    props['sdd'] = float(np.round((np.abs(props['KVDetectorVrt']) + props['KVSourceVrt']) * 10))

    return props

def get_prop_offset(
    prop: str,
    prop_df: Optional[pd.DataFrame] = None,
    cdog_version: Optional[Literal['v2.7', 'v3.0', 'v4.0']] = None,
    ) -> int:
    if prop_df is None and cdog_version is None:
        raise ValueError("Must pass 'prop_df' or 'cdog_version' to get offset.")
    if prop_df is None:
        prop_df = load_prop_df(cdog_version)
        
    row = prop_df[prop_df['prop'] == prop]
    if len(row) == 0:
        raise ValueError(f"Prop '{prop}' not found in metadata props file.")
    offset = row.iloc[0]['byte-offset']
    return offset

def get_prop_pixel(
    base_pixel: Tuple[int, int],
    prop_offset: int,
    size_x: int,
    ) -> Tuple[int, int]:
    # Get the offset pixel.
    # Although there are actually 4 bytes per pixel, only 2 are used and "prop_n_bytes..."
    # only counts used bytes.
    n_bytes_per_pixel = 2
    n_bytes_per_row = size_x * n_bytes_per_pixel
    prop_offset_x = (prop_offset % n_bytes_per_row) // n_bytes_per_pixel
    prop_offset_y = prop_offset // n_bytes_per_row
    if prop_offset_x >= size_x:
        prop_offset_x -= size_x
        prop_offset_y += 1
    prop_pixel = (base_pixel[0] + prop_offset_x, base_pixel[1] + prop_offset_y)

    return prop_pixel

def get_row_stats(
    filepath: str,
    ) -> pd.DataFrame:
    img = PIL.Image.open(filepath)
    data = np.array(img).T

    # How can we determine image vs. metadata rows?
    cols = {
        'row': int,
        'diff-mean': float,
        'diff-min': float,
        'diff-max': float,
        'diff-std': float,
        'diff-90-mean': float,
        'diff-90-min': float,
        'diff-90-max': float,
        'diff-90-std': float,
        'mean': float,
        'min': float,
        'max': float,
        'std': float,
        'n-zeros': int,
    }
    df = pd.DataFrame(columns=cols.keys())

    for i in range(data.shape[1]):
        row_data = data[:, i]
        row_diff = np.diff(row_data)
        row_diff_5, row_diff_95 = np.percentile(row_diff, 5), np.percentile(row_diff, 95)
        row_diff_90 = row_diff[(row_diff > row_diff_5) & (row_diff < row_diff_95)]
        pd_data = {
            'row': i,
            'diff-mean': row_diff.mean(),
            'diff-min': row_diff.min(),
            'diff-max': row_diff.max(),
            'diff-std': row_diff.std(),
            'diff-90-mean': row_diff_90.mean() if len(row_diff_90) > 0 else np.nan,
            'diff-90-min': row_diff_90.min() if len(row_diff_90) > 0 else np.nan,
            'diff-90-max': row_diff_90.max() if len(row_diff_90) > 0 else np.nan,
            'diff-90-std': row_diff_90.std() if len(row_diff_90) > 0 else np.nan,
            'mean': row_data.mean(),
            'min': row_data.min(),
            'max': row_data.max(),
            'std': row_data.std(),
            'n-zeros': len(row_data[row_data == 0]),
        }
        df = append_row(df, pd_data)    

    return df

def infer_cdog_version(
    metadata: np.ndarray
    ) -> Literal['v2.7', 'v3.0', 'v4.0']:
    # Check the 'StartTime' property for something > 2010.
    # Is this robust?
    start_time_offsets = {
        'v2.7': 11148,
        'v3.0': 11164,
        'v4.0': 10988,
    }
    for v, o in start_time_offsets.items():
        try:
            ts = load_timestamp(metadata, prop_offset=o)
        except (OSError, OverflowError, ValueError):
            continue
        if ts.year >= 2010:
            return v

    raise ValueError(f"Couldn't infer CDOG version from passed metadata image.")

# Infers the actual image size - excluding metadata rows.
# Image rows are much smoother than metadata rows. Determine metadata rows by
# high stdev of the diff.
def infer_image_size(
    data: np.ndarray,
    filepath: str,
    ) -> Tuple[int, int]:
    size_x, size_y = data.shape
    if size_x == 1024 and size_y > 768:
        return (1024, 768)
    else:
        raise ValueError(f"'infer_image_size' not implemented for image of size: ({size_x}, {size_y})")

    # for i in range(size_y):
    #     y = size_y - i - 1
    #     row_data = data[:, y]
    #     row_diff = np.diff(row_data)
    #     row_diff_5, row_diff_95 = np.percentile(row_diff, 5), np.percentile(row_diff, 95)
    #     row_diff_90 = row_diff[(row_diff > row_diff_5) & (row_diff < row_diff_95)]
    #     row_diff_90_std = row_diff_90.std()
    #     if row_diff_90_std != np.nan and row_diff_90_std < 200: 
    #         return size_x, y + 1
    # raise ValueError(f"Couldn't infer image size for: {filepath}")

def list_tiff_arcs(
    fraction_path: DirPath,
    ) -> List[int]:
    # Get kV folder.
    kv_dirpath = None
    for f in KV_FOLDERS:
        tmp_dirpath = os.path.join(fraction_path, f)
        if os.path.isdir(tmp_dirpath):
            kv_dirpath = tmp_dirpath
            break
    if kv_dirpath is None:
        raise ValueError(f"No kV subfolder found in '{fraction_path}'. Expected one of: {KV_FOLDERS}")

    files = os.listdir(kv_dirpath)
    files = [f for f in files if f.endswith('.tiff')]
    arcs = [int(f.split('_')[1]) for f in files]
    arcs = to_list(np.unique(arcs))
    return arcs

def list_tiff_fractions(
    pat_path: DirPath,
    ) -> List[int]:
    entries = sorted(os.listdir(pat_path))
    fractions = [int(e.replace('Fx', '')) for e in entries if e.startswith('Fx') and os.path.isdir(os.path.join(pat_path, e))]
    return fractions

def list_tiff_images(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    ) -> List[FilePath]:
    # Get kV folder.
    kv_dirpath = None
    for f in KV_FOLDERS:
        tmp_dirpath = os.path.join(fraction_path, f)
        if os.path.isdir(tmp_dirpath):
            kv_dirpath = tmp_dirpath
            break
    if kv_dirpath is None:
        raise ValueError(f"No kV subfolder found in '{fraction_path}'. Expected one of: {KV_FOLDERS}")

    # Load all images in the arc.
    arcs = arg_to_list(arc, int, literals={'all': lambda: list_tiff_arcs(kv_dirpath)})
    files = [f for f in os.listdir(kv_dirpath) if f.endswith('.tiff')]
    files = list(sorted([f for f in files if int(f.split('_')[1]) in arcs]))
    filepaths = [os.path.join(kv_dirpath, f) for f in files]
    return filepaths

def load_tiff_arc(
    fraction_path: DirPath,
    arc: int,
    **kwargs,
    ) -> Tuple[List[Image2D], List[Dict[str, Any]]]:
    filepaths = list_tiff_images(fraction_path, arc=arc)
    datas, infos = [], []
    for f in filepaths:
        data, info = load_tiff_image(f, **kwargs)
        datas.append(data)
        infos.append(info)
    return datas, infos

def load_tiff_fraction(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    **kwargs,
    ) -> Tuple[List[List[Image2D]], List[List[Dict[str, Any]]]]:
    # Look for kV images.
    kv_dirpath = None
    for f in KV_FOLDERS:
        tmp_dirpath = os.path.join(fraction_path, f)
        if os.path.isdir(tmp_dirpath):
            kv_dirpath = tmp_dirpath
            break
    if kv_dirpath is None:
        raise ValueError(f"No kV subfolder found in '{fraction_path}'. Expected one of: {KV_FOLDERS}")

    # Load all arcs in the fraction.
    arcs = arg_to_list(arc, int, literals={'all': lambda: list_tiff_arcs(kv_dirpath)})
    arc_datas, arc_infos = [], []
    for a in arcs:
        datas, infos = load_tiff_arc(fraction_path, arc=a, **kwargs)
        arc_datas.append(datas)
        arc_infos.append(infos)
    return arc_datas, arc_infos

def load_tiff_patient(
    pat_path: DirPath,
    fraction: int | List[int] | Literal['all'] = 'all',
    **kwargs,
    ) -> Tuple[List[List[List[Image2D]]], List[List[List[Dict[str, Any]]]]]:
    fractions = arg_to_list(fraction, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    all_datas, all_infos = [], []
    for fx in fractions:
        fx_dirpath = os.path.join(pat_path, f"Fx{fx:02d}")
        fx_datas, fx_infos = load_tiff_fraction(fx_dirpath, **kwargs)
        all_datas.append(fx_datas)
        all_infos.append(fx_infos)
    return all_datas, all_infos

# Returns image metadata and pixel data.
def load_tiff_image(
    filepath: FilePath | DirPath,
    arc: int | None = None,
    frame: int | None = None,
    filename_angle: Literal['kv-source', 'kv-detector', 'mv-source', 'mv-detector'] = 'kv-source',
    cdog_version: Optional[str] = None,
    invert_intensities: bool = True,
    load_image: bool = True,
    machine: Literal['elekta', 'varian'] | None = None,
    ) -> Tuple[Image2D | None, Dict[str, Any]]:
    # Get filepath.
    if os.path.isdir(filepath):
        if arc is None or frame is None:
            raise ValueError(f"If 'filepath' is a directory, 'arc' and 'frame' must be passed.")
        filepath = list_tiff_images(filepath, arc=arc)[frame]

    # Add filename metadata.
    info = {}
    filename = os.path.basename(filepath)
    info['dirpath'] = os.path.dirname(filepath)
    info['filepath'] = filepath
    info['filename-arc'] = int(filename.split('_')[1])
    info['filename-frame'] = int(filename.split('_')[2])
    info['filename-angle'] = float(filename.split('_')[-1].replace('.tiff', ''))

    if load_image:
        # Load image.
        img = PIL.Image.open(filepath)
        data = np.array(img).T

        # Figure this out first to get metadata size.
        info['det-size'] = infer_image_size(data, filepath)

        # Extract metadata from image.
        metadata_image = data[:, info['det-size'][1]:]
        info['image'] = metadata_image  # For debugging only.
        base_pixel = get_base_pixel(metadata_image)
        if cdog_version is not None:
            info['cdog-version'] = cdog_version
        else:
            info['cdog-version'] = infer_cdog_version(metadata_image)
        mdata = __load_tiff_metadata(metadata_image, info['cdog-version'])
        info |= mdata

    # Filename angle is ground truth. Calculate other angles based on this.
    # Infer machine from the tiff file?
    if machine is None:
        if load_image:
            machine = 'varian' if 'KVCollimatorX1' in info else 'elekta'
        else:
            raise ValueError(f"Can't infer machine if 'load_image=False', must pass 'machine'.")
    info['kv-source-angle'] = convert_angle(info['filename-angle'], filename_angle, 'kv-source', machine)
    info['kv-detector-angle'] = convert_angle(info['filename-angle'], filename_angle, 'kv-detector', machine)
    info['mv-source-angle'] = convert_angle(info['filename-angle'], filename_angle, 'mv-source', machine)
    info['mv-detector-angle'] = convert_angle(info['filename-angle'], filename_angle, 'mv-detector', machine)

    if not load_image:
        return None, info

    # Extract image data.
    data = data[:, :info['det-size'][1]]

    if invert_intensities:
        data = np.max(data) - data

    return data, info

@bubble_args(
    load_tiff_arc,
    load_tiff_fraction,
    load_tiff_image,
    load_tiff_patient,
)
def load_tiff(
    filepath: FilePath | DirPath,
    arc: int | None = None,
    **kwargs,
    ) -> Tuple:
    if not os.path.isdir(filepath):
        data, info = load_tiff_image(filepath, **kwargs)
    elif os.path.basename(filepath).startswith('Fx'):
        if arc is not None:
            data, info = load_tiff_arc(filepath, arc=arc, **kwargs)
        else:
            data, info = load_tiff_fraction(filepath, **kwargs)
    else:
        data, info = load_tiff_patient(filepath, **kwargs)
    return data, info

def load_tiff_df(
    filepath: str,
    ) -> pd.DataFrame:
    d, m = load_tiff(filepath)
    cols = {
        'prop': str,
        'type': str,
        'val': str,
    }
    df = pd.DataFrame(columns=cols.keys())
    for k, v in m.items():
        if k == 'image':
            continue
        data = {
            'prop': k,
            'type': type(v),
            'val': str(v),
        }
        df = append_row(df, data)
    return df

def load_double(
    metadata: np.ndarray,
    base_flag: int = int('0xCAFE', 16),
    base_pixel: Optional[Tuple[int, int]] = None,
    prop: Optional[str] = None,
    prop_df: Optional[pd.DataFrame] = None,
    prop_offset: Optional[int] = None,
    ) -> float:
    
    if (prop is None or prop_df is None) and prop_offset is None:
        raise ValueError("Either ['prop', 'prop_df'] or 'prop_offset' must be passed.")

    if base_pixel is None:
        base_pixel = get_base_pixel(metadata, base_flag=base_flag)
    if prop_offset is None:
        prop_offset = get_prop_offset(prop, prop_df=prop_df)
    prop_pixel = get_prop_pixel(base_pixel, prop_offset, metadata.shape[0])

    # Load 4 x 2-byte sections of the double and concatenate.
    size_x = metadata.shape[0]
    vals = []
    for i in range(4):
        # Handle wrapping to another row.
        x, y = prop_pixel
        x += i
        if x >= size_x:
            x -= size_x
            y += 1
        val = metadata[x, y]
        vals.append(val)
    vals = np.array([v.astype('<u2') for v in vals])
    val = np.frombuffer(vals.tobytes(), dtype='<f8')[0]
    val = float(val)
    
    return val

def load_prop_df(
    cdog_version: Literal['v2.7', 'v3.0', 'v4.0'],
    ) -> pd.DataFrame:
    basepath = resolve_filepath('files:rtf')
    v_str = cdog_version.replace('v', '').replace('.', '_')
    filepath = os.path.join(basepath, f"XIImagePropertyMapHET{v_str}.csv")
    df = pd.read_csv(filepath, index_col=False, names=['prop', 'datatype', 'byte-offset', 'unsure'], skiprows=7)
    return df

def load_timestamp(
    metadata: np.ndarray,
    base_flag: int = int('0xCAFE', 16),
    base_pixel: Optional[Tuple[int, int]] = None,
    prop: Optional[str] = None,
    prop_df: Optional[pd.DataFrame] = None,
    prop_offset: Optional[int] = None,
    ) -> datetime:

    if (prop is None or prop_df is None) and prop_offset is None:
        raise ValueError("Either ['prop', 'prop_df'] or 'prop_offset' must be passed.")

    if base_pixel is None:
        base_pixel = get_base_pixel(metadata, base_flag=base_flag)
    if prop_offset is None:
        prop_offset = get_prop_offset(prop, prop_df=prop_df)
    prop_pixel = get_prop_pixel(base_pixel, prop_offset, metadata.shape[0])
    
    timestamp_pixel_low_value = metadata[prop_pixel]
    timestamp_pixel_high_value = metadata[prop_pixel[0] + 1, prop_pixel[1]]
    timestamp_remainder_pixel_low_value = metadata[prop_pixel[0] + 2, prop_pixel[1]]
    timestamp_remainder_pixel_high_value = metadata[prop_pixel[0] + 3, prop_pixel[1]]

    # Combine the timestamp parts (correct interpretation)
    timestamp1 = timestamp_pixel_low_value | (timestamp_pixel_high_value << 16)  # Main Unix timestamp
    timestamp2 = timestamp_remainder_pixel_low_value | (timestamp_remainder_pixel_high_value << 16)  # Sub-second precision
    
    # Convert main timestamp to datetime
    main_timestamp = datetime.datetime.fromtimestamp(timestamp1)
    
    # Add sub-second precision (interpreting as nanoseconds)
    subsecond_seconds = timestamp2 / 1000000000.0  # Convert nanoseconds to seconds
    precise_timestamp = datetime.datetime.fromtimestamp(timestamp1 + subsecond_seconds)
    
    # print(f"  Main timestamp: {main_timestamp}")
    # print(f"  With precision: {precise_timestamp}")
    # print(f"  Sub-second: +{timestamp2/1000000:.3f} ms")
    # print(f"  Raw values: 0x{timestamp1:08X}, 0x{timestamp2:08X}")
    # print()

    return precise_timestamp

def plot_tiff_fraction(
    arc_infos: List[List[Dict[str, Any]]],
    arrow_spacing: float = 0.1,
    ax: mpl.axes.Axes | None = None,
    display_angle: str = 'kv-source-angle',
    min_sub_arc_deg: float = 5.0,
    title: str | None = None,
    ) -> None:
    # Build arc angle data from loaded infos.
    arc_data = {}
    arcs = []
    for infos in arc_infos:
        if len(infos) == 0:
            continue
        a = infos[0]['filename-arc']
        arcs.append(a)
        angles = [info[display_angle] for info in infos]

        # Remove consecutive duplicates.
        unique_angles = [angles[0]]
        for ang in angles[1:]:
            if ang != unique_angles[-1]:
                unique_angles.append(ang)

        # Split into sub-arcs at direction changes.
        sub_arcs = []
        if len(unique_angles) >= 2:
            sub_start = unique_angles[0]
            prev_cw = unique_angles[1] > unique_angles[0]
            for j in range(2, len(unique_angles)):
                if unique_angles[j] == unique_angles[j - 1]:
                    continue
                curr_cw = unique_angles[j] > unique_angles[j - 1]
                if curr_cw != prev_cw:
                    sub_arcs.append((sub_start, unique_angles[j - 1], prev_cw))
                    sub_start = unique_angles[j - 1]
                    prev_cw = curr_cw
            sub_arcs.append((sub_start, unique_angles[-1], prev_cw))
        else:
            sub_arcs.append((unique_angles[0], unique_angles[0], True))

        # Merge sub-arcs that are shorter than min_sub_arc_deg (noise/jitter).
        merged = True
        while merged:
            merged = False
            for i in range(len(sub_arcs)):
                span = abs(sub_arcs[i][1] - sub_arcs[i][0])
                if span < min_sub_arc_deg and len(sub_arcs) > 1:
                    if i + 1 < len(sub_arcs):
                        nxt = sub_arcs[i + 1]
                        sub_arcs[i + 1] = (sub_arcs[i][0], nxt[1], nxt[2])
                        sub_arcs.pop(i)
                    else:
                        prv = sub_arcs[i - 1]
                        sub_arcs[i - 1] = (prv[0], sub_arcs[i][1], prv[2])
                        sub_arcs.pop(i)
                    merged = True
                    break

        arc_data[a] = sub_arcs

    # Build full range of arc indices so missing arcs leave empty rings.
    all_arcs = list(range(max(arcs) + 1))
    arc_ring_counts = {}
    for a in all_arcs:
        arc_ring_counts[a] = len(arc_data[a]) if a in arc_data else 1

    # Plot concentric arcs on a polar axes.
    cb_palette = sns.color_palette('colorblind')
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    ring_width = 0.15
    sub_ring_width = ring_width / 3
    arc_gap = 0.05
    sub_ring_gap = 0.01
    inner_radius = 0.4

    # Compute the starting radius for each arc.
    arc_radii = {}
    current_radius = inner_radius
    for a in all_arcs:
        arc_radii[a] = current_radius
        n_rings = arc_ring_counts[a]
        if n_rings == 1:
            current_radius += ring_width + arc_gap
        else:
            current_radius += n_rings * (sub_ring_width + sub_ring_gap) - sub_ring_gap + arc_gap

    for a in all_arcs:
        if a not in arc_data:
            continue

        sub_arcs = arc_data[a]
        color = cb_palette[(a - 1) % len(cb_palette)]
        base_radius = arc_radii[a]
        n_sub = len(sub_arcs)

        for si, (start_angle, end_angle, clockwise) in enumerate(sub_arcs):
            if clockwise:
                theta_span_deg = (end_angle - start_angle) % 360
            else:
                theta_span_deg = (start_angle - end_angle) % 360
            theta_start = np.deg2rad(start_angle)
            theta_span = np.deg2rad(theta_span_deg)

            if n_sub == 1:
                rw = ring_width
                radius = base_radius
            else:
                rw = sub_ring_width
                radius = base_radius + si * (sub_ring_width + sub_ring_gap)

            theta_mid = theta_start + theta_span / 2 if clockwise else theta_start - theta_span / 2
            if si == 0:
                overall_start = sub_arcs[0][0]
                overall_end = sub_arcs[-1][1]
                label = f"{a} ({overall_start:.1f}° → {overall_end:.1f}°"
                if n_sub > 1:
                    label += f", {n_sub} segs"
                label += ")"
            else:
                label = None
            ax.bar(
                x=theta_mid,
                height=rw,
                width=theta_span,
                bottom=radius,
                color=color,
                alpha=0.7,
                label=label,
                edgecolor='white',
                linewidth=0.5,
            )

            # Add directional arrows.
            arrow_radius = radius + rw / 2
            angular_spacing = np.rad2deg(arrow_spacing / arrow_radius)
            delta = np.deg2rad(2) if clockwise else np.deg2rad(-2)
            n_arrows = int(theta_span_deg / angular_spacing)
            for i in range(n_arrows):
                if clockwise:
                    a_deg = start_angle + i * angular_spacing
                else:
                    a_deg = start_angle - i * angular_spacing
                a_rad = np.deg2rad(a_deg)
                ax.annotate(
                    '',
                    xy=(a_rad + delta, arrow_radius),
                    xytext=(a_rad - delta, arrow_radius),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                )

    # Style the polar plot.
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, current_radius + 0.1)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{a}°" for a in range(0, 360, 30)])
    if title is None:
        # Infer fraction name from dirpath.
        if len(arc_infos) > 0 and len(arc_infos[0]) > 0:
            parts = os.path.normpath(arc_infos[0][0]['dirpath']).split(os.sep)
            title = parts[-2]
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()

def plot_tiff_patient(
    info: List[List[List[Dict[str, Any]]]],
    n_cols: int = 4,
    **kwargs,
    ) -> None:
    n_fractions = len(info)
    n_rows = int(np.ceil(n_fractions / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': 'polar'}, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, arc_infos in enumerate(info):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        plot_tiff_fraction(arc_infos, ax=ax, **kwargs)

    # Hide unused axes.
    for i in range(n_fractions, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_tiff_image(
    data: Image2D,
    info: Dict[str, Any],
    ax: mpl.axes.Axes | None = None,
    hist_eq: bool = False,
    normalise: bool = False,
    other_info: List[Dict[str, Any]] | None = None,
    return_image: bool = False,
    show_hist: bool = False,
    show_waveform: bool = True,
    title_fontsize: float = 10,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ) -> Optional[PIL.Image.Image | List[PIL.Image.Image]]:
    if show_waveform:
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
        ax = fig.add_subplot(gs[:, 0])
        amp_ax = fig.add_subplot(gs[0, 1])
        phase_ax = fig.add_subplot(gs[1, 1])
        show = True
    elif show_hist:
        _, axs = plt.subplots(1, 2, figsize=(12, 4))
        ax, hist_ax = axs
        show = True
    elif ax is None:
        _, ax = plt.subplots()
        show = True
    else:
        show = False

    # Normalise data.
    aspect = info['det-size'][1] / info['det-size'][0]
    if normalise:
        data = (data - data.min()) / (data.max() - data.min())
    if hist_eq:
        data = exposure.equalize_hist(data)

    # Plot slice.
    ax.imshow(data.T, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)

    # Plot histogram.
    if show_hist:
        hist_ax.hist(data.flatten(), bins=50, color='gray')

    # Plot waveforms.
    if show_waveform:
        # Plot other slices' amplitude/phase.
        cb_palette = sns.color_palette('colorblind')
        if other_info is not None:
            base_color = cb_palette[1]
            cmap = sns.light_palette(base_color, as_cmap=True)
            n = len(other_info)
            for idx, i in enumerate(other_info):
                color = cmap((idx + 1) / (n + 1))
                angle = i['kv-source-angle']
                amp_ax.scatter(angle, i['MMAmplitude0'], color=color, zorder=0)
                phase_ax.scatter(angle, i['MMPhase0'], color=color, zorder=0)

        # Plot current slice's amplitude/phase.
        angle = info['kv-source-angle']
        amp_ax.scatter(angle, info['MMAmplitude0'], color=cb_palette[0], zorder=1)
        phase_ax.scatter(angle, info['MMPhase0'], color=cb_palette[0], zorder=1)

        amp_ax.set_xlim(0, 360)
        amp_ax.set_ylabel('MMAmplitude0')
        amp_ax.set_title('Amplitude', fontsize=title_fontsize)
        phase_ax.set_xlim(0, 360)
        phase_ax.set_ylim(0, 360)
        phase_ax.set_xlabel('kV source angle [degrees]')
        phase_ax.set_ylabel('MMPhase0')
        phase_ax.set_title('Phase', fontsize=title_fontsize)

    # Add annotation.
    cdog_version = info['cdog-version']
    title = f"CDOG ({cdog_version}) TIFF image ({info['det-size'][0]} x {info['det-size'][1]})\n\
Filename - arc: {info['filename-arc']}, frame: {info['filename-frame']}, angle: {info['filename-angle']}\n\
Imin/max: {data.min()}/{data.max()}, Vmin/max: {vmin}/{vmax}\n\
MV source/det: {info['mv-source-angle']:.1f}°/{info['mv-detector-angle']:.1f}°\n\
kV source/det: {info['kv-source-angle']:.1f}°/{info['kv-detector-angle']:.1f}°"
    ax.set_title(title, fontsize=title_fontsize)
    # plt.axis('off')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(f"LR [{info['det-spacing'][0]}mm]")
    ax.set_ylabel(f"SI [{info['det-spacing'][1]}mm]")

    if return_image:
        buf = io.BytesIO()
        ax.figure.savefig(buf, format='png', bbox_inches='tight')
        plt.close(ax.figure)
        buf.seek(0)
        return PIL.Image.open(buf).copy()
    elif show:
        plt.show()

def plot_gif(
    plot_fn: Callable,
    datas: List[Any],
    infos: List[Dict[str, Any]],
    frame_time: float = 0.5,
    load: bool = True,
    loop: bool = True,
    n_frames: int | None = None,
    overwrite: bool = False,
    pause_time: float = 5,
    width: int = 1000,
    **kwargs,
    ) -> None:
    # Load from disk.
    arc = infos[0]['filename-arc']
    dirpath = infos[0]['dirpath']
    savepath = os.path.join(dirpath, f"arc_{arc}.apng")
    if load and not overwrite and os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            display(IPythonImage(data=f.read(), format='png', width=width))
        return

    # Generate frames.
    png_images = []
    total = len(datas) if n_frames is None else min(len(datas), n_frames)
    for i, (d, inf) in tqdm(enumerate(zip(datas, infos)), total=total, desc="Creating GIF frames"):
        png_image = plot_fn(d, inf, return_image=True, **kwargs)
        png_images.append(png_image)
        if n_frames is not None and i + 1 >= n_frames:
            break

    # Save animated PNG (avoids GIF's 256-colour palette limitation).
    frames = png_images
    frames_per_second = 1 / frame_time
    frames = frames + [frames[-1]] * int(pause_time / frame_time)
    imageio.mimsave(savepath, frames, fps=frames_per_second, loop=0 if loop else None)

    with open(savepath, 'rb') as f:
        display(IPythonImage(data=f.read(), format='png', width=width))

def plot_tiff_gif(
    datas: List[Image2D] | None = None,
    infos: List[Dict[str, Any]] | None = None,
    **kwargs,
    ) -> None:
    def _plot_tiff_frame(d, inf, **kw):
        other_info = [x for x in infos if x is not inf]
        return plot_tiff_image(d, inf, other_info=other_info, **kw)

    plot_gif(_plot_tiff_frame, datas, infos, **kwargs)

def plot_tiff_hist(
    filepath: str,
    **kwargs,
    ) -> None:
    data, _ = load_tiff(filepath)
    plt.hist(data)
    plt.show()
