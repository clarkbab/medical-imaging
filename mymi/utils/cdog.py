import datetime
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

from .args import arg_to_list
from .io import resolve_filepath
from .pandas import append_row

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

def get_metadata(
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
    props['kv-source-angle'] = float(np.round(props['KVSourceRtn'] % 360, decimals=3))
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
        ts = load_timestamp(metadata, prop_offset=o)
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
    dirpath: DirPath,
    ) -> List[str]:
    tiff_files = list(sorted([f for f in os.listdir(dirpath) if f.endswith('.tiff')]))
    arcs = list([str(a) for a in np.unique(['_'.join(f.split('_')[:2]) for f in tiff_files])])
    return arcs

def load_tiff(
    filepath: FilePath | DirPath,
    **kwargs,
    ) -> Tuple[List[Slice], List[Dict[str, Any]]]:
    if os.path.isdir(filepath):
        data, info = load_tiff_dirpath(filepath, **kwargs)
    else:
        data, info = load_tiff_filepath(filepath, **kwargs)
    return data, info

def load_tiff_dirpath(
    dirpath: DirPath,
    angle_range: Tuple[float | None, float | None] = (None, None),
    arc: int | None = 0,
    n_angles: int | None = None,
    n_frames: int | None = None,
    **kwargs,
    ) -> Tuple[List[Slice], List[Dict[str, Any]]]:
    # Get tiff files and angles (by filename).
    files = os.listdir(dirpath)
    tiff_files = list(sorted([f for f in files if f.endswith('.tiff')]))
    tiff_angles = np.array([float(f.split('_')[-1].replace('.tiff', '')) for f in tiff_files])

    # Filter tiff files by arc.
    arcs = list([str(a) for a in np.unique(['_'.join(f.split('_')[:2]) for f in tiff_files])])
    arc = arcs[arc]
    tiff_angles = [a for f, a in zip(tiff_files, tiff_angles) if f.startswith(arc)]
    tiff_files = [f for f in tiff_files if f.startswith(arc)]

    # Filter by angle range.
    if angle_range[0] is not None:
        indices = np.where(np.array(tiff_angles) >= angle_range[0])[0]
        tiff_angles = [tiff_angles[i] for i in indices]
        tiff_files = [tiff_files[i] for i in indices]
    if angle_range[1] is not None:
        indices = np.where(np.array(tiff_angles) < angle_range[1])[0]
        tiff_angles = [tiff_angles[i] for i in indices]
        tiff_files = [tiff_files[i] for i in indices]

    # Only include subset of angles - split evenly across the range.
    if n_angles is not None:
        # tiff_angle_range = np.abs(tiff_angles[-1] - tiff_angles[0])     # Assumes arm can't reverse within an arc?
        # print(tiff_angle_range)
        # print(plot_freq)
        plot_freq = np.max([len(tiff_files) // n_angles, 1])
        tiff_files = [f for i, f in enumerate(tiff_files) if i % plot_freq == 0] 

    # Get first n frames.
    if n_frames is not None:
        tiff_files = tiff_files[:n_frames]

    # Load tiff files.
    filepaths = [os.path.join(dirpath, f) for f in tiff_files]
    datas, infos = [], []
    for f in tqdm(filepaths, desc="Loading TIFF files"):
        data, info = load_tiff_filepath(f, **kwargs)
        datas.append(data)
        infos.append(info)
    return datas, infos

# Returns image metadata and pixel data.
def load_tiff_filepath(
    filepath: str,
    invert_intensities: bool = True,
    cdog_version: Optional[str] = None,
    ) -> Tuple[Slice, Dict[str, Any]]:
    # Load image.
    img = PIL.Image.open(filepath)
    data = np.array(img).T

    # Get TIFF metadata.
    # We don't really need this, just checks the data type for future calcs.
    # tiff_metadata = load_tiff_metadata(img)

    # Build our own metadata.
    metadata = {}
    filename = os.path.basename(filepath)
    metadata['filepath'] = filepath
    metadata['arc'] = '_'.join(filename.split('_')[:2])
    metadata['frame'] = filename.split('_')[2]
    metadata['angle'] = filename.split('_')[-1].replace('.tiff', '')
    metadata['det-size'] = infer_image_size(data, filepath)

    # Extract metadata from .
    metadata_image = data[:, metadata['det-size'][1]:]
    metadata['image'] = metadata_image  # For debugging only.
    base_pixel = get_base_pixel(metadata_image)
    if cdog_version is not None:
        metadata['cdog-version'] = cdog_version
    else:
        metadata['cdog-version'] = infer_cdog_version(metadata_image)
    mdata = get_metadata(metadata_image, metadata['cdog-version'])
    metadata |= mdata

    # Extract image data.
    image_data = data[:, :metadata['det-size'][1]]

    if invert_intensities:
        image_data = np.max(image_data) - image_data

    return image_data, metadata

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

def plot_arc_ranges(
    dirpath: DirPath,
    arc: int | Literal['all'] = 'all',
    ) -> None:
    # Get all arc names from directory.
    files = os.listdir(dirpath)
    tiff_files = list(sorted([f for f in files if f.endswith('.tiff')]))
    all_arcs = list(np.unique(['_'.join(f.split('_')[:2]) for f in tiff_files]))

    if arc != 'all':
        all_arcs = [all_arcs[arc]]

    # For each arc, get min/max angle from filenames.
    arc_ranges = []
    for a in all_arcs:
        arc_files = [f for f in tiff_files if f.startswith(a)]
        angles = [float(f.split('_')[-1].replace('.tiff', '')) for f in arc_files]
        arc_ranges.append((a, min(angles), max(angles)))

    # Plot concentric arcs on a polar axes.
    cb_palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))

    ring_width = 0.15
    inner_radius = 0.4

    for idx, (name, start_angle, end_angle) in enumerate(arc_ranges):
        color = cb_palette[idx % len(cb_palette)]
        radius = inner_radius + idx * (ring_width + 0.05)

        # Convert degrees to radians (0° at top, clockwise).
        theta_start = np.deg2rad(start_angle)
        theta_end = np.deg2rad(end_angle)

        # Draw arc as a filled polar bar.
        theta_span = theta_end - theta_start
        ax.bar(
            x=theta_start + theta_span / 2,
            height=ring_width,
            width=theta_span,
            bottom=radius,
            color=color,
            alpha=0.7,
            label=f"{name} ({start_angle:.1f}° - {end_angle:.1f}°)",
            edgecolor='white',
            linewidth=0.5,
        )

    # Style the polar plot.
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, inner_radius + len(arc_ranges) * (ring_width + 0.05) + 0.1)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 30)))
    ax.set_xticklabels([f"{a}°" for a in range(0, 360, 30)])
    ax.set_title('Arc Angular Ranges', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    plt.tight_layout()
    plt.show()

def plot_tiff(
    data: Slice,
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
Arc: {info['arc']}, frame: {info['frame']}, angle: {info['angle']}\n\
Imin/max: {data.min()}/{data.max()}, Vmin/max: {vmin}/{vmax}\n\
MV source angle: {info['GantryRtn']:.3f}\n\
kV source/det. angle: {info['kv-source-angle']:.3f}/{info['KVDetectorRtn']:.3f}"
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

def plot_tiff_gif(
    data: List[Slice] | None = None,
    info: List[Dict[str, Any]] | None = None,
    frame_time: float = 0.5,
    loop: bool = True,
    n_frames: int | None = None,
    overwrite: bool = False,
    pause_time: float = 5,
    savepath: FilePath | None = None,
    width: int = 1000,
    **kwargs,
    ) -> None:
    if savepath is not None and os.path.exists(savepath) and not overwrite:
        with open(savepath, 'rb') as f:
            display(IPythonImage(data=f.read(), format='png', width=width))
        return

    if savepath is None:
        savepath = tempfile.NamedTemporaryFile(suffix='.apng', delete=False).name

    # Get tiff images.
    png_images = []
    for i, (d, inf) in tqdm(enumerate(zip(data, info)), total=len(data), desc="Creating GIF frames"):
        other_info = info[:i] + info[i+1:]
        png_image = plot_tiff(d, inf, other_info=other_info, return_image=True, **kwargs)
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

def plot_tiff_hist(
    filepath: str,
    **kwargs,
    ) -> None:
    data, _ = load_tiff(filepath)
    plt.hist(data)
    plt.show()
