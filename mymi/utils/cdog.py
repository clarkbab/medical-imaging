import datetime
import dicomset as ds
import io
from PIL import Image as PILImage
from dicomset.utils import filter_lists, sort_lists
from importlib.metadata import metadata
from multiprocessing.util import info
from dicomset.typing import *
from dicomset.utils import bubble_args, logger, to_list, hist_eq as hist_eq_fn, save_numpy, save_csv, load_numpy, load_csv

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
from tqdm.auto import tqdm
from typing import *

from mymi.typing import *

from .projections import convert_angles, plot_projection
from .args import arg_to_list
from .io import resolve_filepath
from .pandas import append_row
from .plotting import plot_gif

KV_FOLDERS = ['KIM-KV', 'kV']

# CT windowing presets: (width, level).
WINDOW_PRESETS = {
    'bone': (1800, 400),
    'brain': (80, 40),
    'liver': (150, 30),
    'lung': (1500, -600),
    'mediastinum': (350, 50),
    'tissue': (400, 50),
}

def __resolve_window(
    window: Window | None,
    vmin: float | None,
    vmax: float | None,
    ) -> tuple[float | None, float | None]:
    if window is not None:
        assert vmin is None, "vmin must be None if window is specified."
        assert vmax is None, "vmax must be None if window is specified."
    if window is None:
        return vmin, vmax
    if isinstance(window, str):
        if window not in WINDOW_PRESETS:
            raise ValueError(f"Unknown window preset '{window}'. Expected one of {list(WINDOW_PRESETS.keys())}.")
        width, level = WINDOW_PRESETS[window]
    else:
        width, level = window
    vmin = level - width / 2
    vmax = level + width / 2
    return vmin, vmax

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

def __load_metadata(
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
    kv_dirpath = os.path.join(fraction_path, 'kV_proc')
    files = os.listdir(kv_dirpath)
    files = [f for f in files if f.startswith('arc_') and f.endswith('.npz')]
    arcs = [int(f.split('_')[1].replace('.npz', '')) for f in files]
    arcs = list(sorted(np.unique(arcs)))
    return arcs

def list_raw_arcs(
    fraction_path: DirPath,
    ) -> List[int]:
    # Get kV folder.
    kv_dirpath = get_kv_path(fraction_path)

    files = os.listdir(kv_dirpath)
    files = [f for f in files if f.endswith('.tiff')]
    arcs = [int(f.split('_')[1]) for f in files]
    arcs = to_list(np.unique(arcs))
    return arcs

def list_tiff_fractions(
    pat_path: DirPath,
    ) -> List[str]:
    fractions = [f for f in os.listdir(pat_path) if f.lower().startswith('fx')]
    fractions = [int(f.lower().replace('fx', '')) for f in fractions]
    return fractions

def list_tiff(
    fraction_path: DirPath,
    arc: int,
    ) -> List[FilePath]:
    kv_dirpath = get_kv_path(fraction_path)

    # Load all images in the arc.
    tiff_files = list(sorted([f for f in os.listdir(kv_dirpath) if f.endswith('.tiff')]))
    arcs = [int(f.split('_')[1]) for f in tiff_files]
    frames = [int(f.split('_')[2]) for f in tiff_files]
    tiff_files, arcs, frames = filter_lists([tiff_files, arcs, frames], lambda taf: taf[1] == arc)
    tiff_files, arcs, frames = sort_lists([tiff_files, arcs, frames], key=lambda taf: taf[2])
    tiff_filepaths = [os.path.join(kv_dirpath, f) for f in tiff_files]
    return tiff_filepaths

def plot_arc_shroud(
    shroud: Image2D,
    aspect: float = 0.1,
    ax: mpl.axes.Axes | None = None,
    figsize: Tuple[float, float] | None = None,
    frames: List[int] | None = None,
    signals: np.ndarray | None = None,  # Could be 2d, first dimension is for multiple signals.
    **kwargs,
    ) -> None:
    if ax is None:
        if figsize is not None:
            plt.figure(figsize=figsize)
        ax = plt.gca()
        show = True
    else:
        show = False
    if signals is not None:
        signals = np.array(signals)
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        assert signals.shape[1] == shroud.shape[0], f"Length of signal must match number of frames in shroud. Got len(signal)={signals.shape[1]}, shroud.shape[0]={shroud.shape[0]}"
    ax.imshow(shroud.T, aspect=aspect, cmap='gray', **kwargs)
    if signals is not None:
        signal_palette = sns.color_palette('colorblind', n_colors=signals.shape[0])
        for i, s in enumerate(signals):
            # Normalise signal to shroud height.
            # s = (s - np.min(s)) / (np.max(s) - np.min(s)) * shroud.shape[1]
            ax.plot(s, color=signal_palette[i], linewidth=2)
            # ax.scatter(range(len(s)), s, color=signal_palette[i], linewidth=2)
    # Plot scatter points at the top for each frame if provided
    if frames is not None and len(frames) > 0:
        cb_palette = sns.color_palette('colorblind')
        n_colors = len(cb_palette)
        y_top = 0  # Top row of the image (after transpose)
        for i, frame in enumerate(frames):
            color = cb_palette[i % n_colors]
            ax.scatter(frame, y_top, color=color, s=40, marker='o', edgecolor='black', zorder=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if show:
        plt.show()

def plot_fraction_shrouds(
    shrouds: List[Image2D],
    infos: List[Dict[str, Any]],
    aspect: float = 0.1,
    axs: List[mpl.axes.Axes] | None = None,
    signals: List[np.ndarray] | None = None,
    figsize: Tuple[float, float] | None = None,
    **kwargs,
    ) -> None:
    n_rows = len(shrouds)
    if axs is None:
        _, axs = plt.subplots(n_rows, 1, sharex=True, figsize=figsize, gridspec_kw={'hspace': 0}, squeeze=False)
        show = True
    else:
        assert len(axs) == n_rows, f"Number of axes ({len(axs)}) must match number of shrouds ({n_rows})."
        show = False
    for i, s in enumerate(shrouds):
        signal = signals[i] if signals is not None else None
        plot_arc_shroud(s, aspect=aspect, ax=axs[i, 0], signals=signal, **kwargs)
        if i < len(shrouds) - 1:
            axs[i, 0].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        axs[i, 0].tick_params(axis='y', which='both', left=False, labelleft=False)
        axs[i, 0].set_ylabel(f'Arc {infos[i]["arc"]}')
    fraction = infos[0]['fraction']
    axs[0, 0].set_title(f"Fraction {fraction}")
        
    if show:
        plt.tight_layout()
        plt.show()

def create_tiff_arc_shroud(
    fraction_path: DirPath,
    arc: int,
    hist_eq: bool = True,
    n_frames: int | None = None,
    raw: bool = False,
    show_progress: bool = True,
    ) -> None:
    if raw:
        data, _ = load_tiff_arc(fraction_path, arc, n_frames=n_frames, raw=True, show_progress=show_progress)
    else:
        data, _ = load_tiff_arc(fraction_path, arc, n_frames=n_frames)
    n_frames = data.shape[0]
    # Histogram equalisation so that frames are more consistent across the arc.
    if hist_eq:
        for i in tqdm(range(n_frames), desc=f"Normalising frames", disable=not show_progress):
            data[i] = hist_eq_fn(data[i])
    shroud = data.mean(axis=1)
    if raw:
        if hist_eq:
            filepath = os.path.join(fraction_path, 'kV_proc', f'raw_arc_{arc}_shroud.npz')
        else:
            filepath = os.path.join(fraction_path, 'kV_proc', f'raw_arc_{arc}_shroud_nohist.npz')
    else:
        if hist_eq:
            filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_shroud.npz')
        else:
            filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_shroud_nohist.npz')
    save_numpy(shroud, filepath)

def load_tiff_arc_shroud(
    fraction_path: DirPath,
    arc: int,
    hist_eq: bool = True,
    load_data: bool = True,
    raw: bool = False,
    ) -> Tuple[Image2D | None, Dict[str, Any]]:
    if load_data:
        if raw:
            if hist_eq:
                filepath = os.path.join(fraction_path, 'kV_proc', f'raw_arc_{arc}_shroud.npz')
            else:
                filepath = os.path.join(fraction_path, 'kV_proc', f'raw_arc_{arc}_shroud_nohist.npz')
        else:
            if hist_eq:
                filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_shroud.npz')
            else:
                filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_shroud_nohist.npz')
        data = load_numpy(filepath)
    else:
        data = None
    fraction = int(os.path.basename(fraction_path).lower().replace('fx', ''))
    info = dict(arc=arc, fraction=fraction)
    return data, info

def load_tiff_arc(
    fraction_path: DirPath,
    arc: int,
    load_data: bool = True,
    n_frames: int | None = None,
    raw: bool = False,
    show_progress: bool = True,
    **kwargs,
    ) -> Tuple[BatchImage2D | None, pd.DataFrame]:
    if raw:
        filepaths = list_tiff(fraction_path, arc=arc)
        if n_frames is not None:
            filepaths = filepaths[:n_frames]
        datas, infos = [], []
        for f in tqdm(filepaths, desc=f"Loading raw frames", disable=not show_progress):
            try: 
                data, info = load_tiff(f, **kwargs)
            except PIL.UnidentifiedImageError as e:
                logger.warn(f"Couldn't load frame '{f}'. Skipping. Error: {e}")
                continue
            datas.append(data)
            infos.append(info)
        datas = np.stack(datas, axis=0)
    else:
        if load_data:
            filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}.npz')
            datas = load_numpy(filepath).astype(np.float32)
        else:
            datas = None
        filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_info.csv')
        infos = load_csv(filepath, eval_cols=['det-size', 'det-spacing', 'det-offset'])
        if n_frames is not None:
            datas = datas[:n_frames]
            infos = infos.iloc[:n_frames]
    return datas, infos

def process_tiff_arc(
    fraction_path: DirPath,
    arc: int,
    n_frames: int | None = None,
    show_progress: bool = True,
    **kwargs,
    ) -> Tuple[BatchImage2D, List[Dict[str, Any]]]:
    data, info = load_tiff_arc(fraction_path, arc, n_frames=n_frames, raw=True, show_progress=show_progress, **kwargs)

    # Apply histogram equalisation.
    n_frames = data.shape[0]
    # for i in tqdm(range(n_frames), desc=f"Frames", disable=not show_progress):
    #     data[i] = hist_eq_fn(data[i])
    filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}.npz')
    save_numpy(data, filepath)

    # Filter out 'image'.
    remove_keys = ['image']
    filt_info = []
    for i in info:
        i = dict([(k, v) for k, v in i.items() if k not in remove_keys])
        filt_info.append(i)

    # Create dataframe.
    df = pd.DataFrame(filt_info)

    # Add other info to frame.
    df['arc'] = arc
    df['fraction'] = int(os.path.basename(fraction_path).lower().replace('fx', ''))

    filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_info.csv')
    save_csv(df, filepath)

def load_tiff_patient_shrouds(
    pat_path: DirPath,
    fraction: int | List[int] | Literal['all'] = 'all',
    **kwargs,
    ) -> Tuple[List[BatchImage2D], List[List[pd.DataFrame]]]:
    fractions = arg_to_list(fraction, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    shrouds = []
    infos = []
    for f in fractions:
        fx_dirpath = os.path.join(pat_path, f"Fx{f:02d}")
        shroud, info = load_tiff_fraction_shrouds(fx_dirpath, **kwargs)
        shrouds.append(shroud)
        infos.append(info)
    return shrouds, infos

def load_tiff_fraction_shrouds(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    **kwargs,
    ) -> Tuple[List[Image2D], List[Dict[str, Any]]]:
    arcs = arg_to_list(arc, int, literals={'all': lambda: list_tiff_arcs(fraction_path)})
    shrouds = []
    infos = []
    for a in arcs:
        shroud, info = load_tiff_arc_shroud(fraction_path, a)
        shrouds.append(shroud)
        infos.append(info)
    return shrouds, infos

def load_tiff_fraction(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    raw: bool = False,
    show_progress: bool = True,
    **kwargs,
    ) -> Tuple[List[BatchImage2D], List[pd.DataFrame]]:
    if raw:
        kv_dirpath = get_kv_path(fraction_path)

        # Load all arcs in the fraction.
        arcs = arg_to_list(arc, int, literals={'all': lambda: list_raw_arcs(fraction_path)})
        datas, infos = [], []
        for a in tqdm(arcs, desc=f"Loading raw arcs", disable=not show_progress):
            data, info = load_tiff_arc(fraction_path, a, raw=True, show_progress=False, **kwargs)
            datas.append(data)
            infos.append(info)
    else:
        arcs = arg_to_list(arc, int, literals={'all': lambda: list_tiff_arcs(fraction_path)})
        datas, infos = [], []
        for a in arcs:
            data, info = load_tiff_arc(fraction_path, a, **kwargs)
            datas.append(data)
            infos.append(info)
    return datas, infos

def process_fraction(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> Tuple[List[BatchImage2D], List[List[Dict[str, Any]]]]:
    # Look for kV images.
    kv_dirpath = get_kv_path(fraction_path)

    # Make sure all arcs contain consecutive frames.
    assert_arc_jumps(fraction_path)

    # Process all arcs in the fraction. 
    arcs = arg_to_list(arc, int, literals={'all': lambda: list_raw_arcs(fraction_path)})
    for a in tqdm(arcs, desc=f"Processing raw arcs", disable=not show_progress):
        process_tiff_arc(fraction_path, arc=a, show_progress=show_progress, **kwargs)

def load_tiff_patient(
    pat_path: DirPath,
    fraction: int | List[int] | Literal['all'] = 'all',
    **kwargs,
    ) -> Tuple[List[BatchImage2D], List[List[pd.DataFrame]]]:
    fractions = arg_to_list(fraction, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    datas, infos = [], []
    for f in fractions:
        fx_dirpath = os.path.join(pat_path, f"Fx{f:02d}")
        data, info = load_tiff_fraction(fx_dirpath, **kwargs)
        datas.append(data)
        infos.append(info)
    return datas, infos

def process_tiff_patient(
    pat_path: DirPath,
    fraction: int | List[int] | Literal['all'] = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> Tuple[List[BatchImage2D], List[List[List[Dict[str, Any]]]]]:
    fractions = arg_to_list(fraction, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    for fx in tqdm(fractions, desc="Processing raw fractions", disable=not show_progress):
        fx_dirpath = os.path.join(pat_path, f"Fx{fx:02d}")
        process_fraction(fx_dirpath, show_progress=show_progress, **kwargs)

# Returns image metadata and pixel data.
def load_tiff(
    filepath: FilePath | DirPath,
    arc: int | None = None,
    frame: int | None = None,
    filename_angle: Literal['kv-source', 'kv-detector', 'mv-source', 'mv-detector'] = 'kv-source',
    # Blank refers to empty metadata.
    cdog_version: str | Literal['blank'] | None = None,
    flip_lr: bool = True,
    invert_intensity: bool = True,
    load_data: bool = True,
    machine: Literal['elekta', 'varian'] | None = None,
    ) -> Tuple[Image2D | None, pd.Series]:
    # with run_once():
    # logger.warn(f"Loading frames with 'filename_angle={filename_angle}' and 'machine={machine}'.")
    # Get filepath.
    if os.path.isdir(filepath):
        if arc is None or frame is None:
            raise ValueError(f"If 'filepath' is a directory, 'arc' and 'frame' must be passed.")
        filepath = list_tiff(filepath, arc=arc)[frame]

    # Add filename metadata.
    info = {}
    filename = os.path.basename(filepath)
    info['dirpath'] = os.path.dirname(filepath)
    info['filepath'] = filepath
    info['filename-arc'] = int(filename.split('_')[1])
    info['filename-frame'] = int(filename.split('_')[2])
    info['filename-angle'] = float(filename.split('_')[-1].replace('.tiff', ''))

    if load_data:
        # Load image.
        img = PIL.Image.open(filepath)
        data = np.array(img).T

        # Figure this out first to get metadata size.
        info['det-size'] = infer_image_size(data, filepath)

        # Extract metadata from image.
        metadata_image = data[:, info['det-size'][1]:]
        info['image'] = metadata_image  # For debugging only.
        if cdog_version is not None:
            info['cdog-version'] = cdog_version
        else:
            info['cdog-version'] = infer_cdog_version(metadata_image)
        if info['cdog-version'] != 'blank':
            mdata = __load_metadata(metadata_image, info['cdog-version'])
            info |= mdata

    # Filename angle is ground truth. Calculate other angles based on this.
    # Infer machine from the tiff file?
    if machine is None:
        if load_data:
            machine = 'varian' if 'KVCollimatorX1' in info else 'elekta'
        else:
            raise ValueError(f"Can't infer machine if 'load_data=False', must pass 'machine'.")
    info['machine'] = machine
    info['kv-source-angle'] = convert_angles(info['filename-angle'], filename_angle, 'kv-source', machine)
    info['kv-detector-angle'] = convert_angles(info['filename-angle'], filename_angle, 'kv-detector', machine)
    info['mv-source-angle'] = convert_angles(info['filename-angle'], filename_angle, 'mv-source', machine)
    info['mv-detector-angle'] = convert_angles(info['filename-angle'], filename_angle, 'mv-detector', machine)

    if not load_data:
        return None, info

    # Extract image data.
    data = data[:, :info['det-size'][1]].astype(np.float32)

    if invert_intensity:
        data = np.max(data) - data

    if flip_lr:
        data = np.flip(data, axis=0)
    data = np.flip(data, axis=1)

    info = pd.Series(info)

    return data, info

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
    
    # Add sub-second precision (interpreting as nanoseconds)
    subsecond_seconds = timestamp2 / 1000000000.0  # Convert nanoseconds to seconds
    try:
        precise_timestamp = datetime.datetime.fromtimestamp(timestamp1 + subsecond_seconds)
    except OSError as e:
        logger.error(f"Error converting timestamp: {e}. Returning main timestamp without sub-second precision.")
        try:
            precise_timestamp = datetime.datetime.fromtimestamp(timestamp1)
        except OSError as e:
            logger.error(f"Error converting main timestamp: {e}. Returning Unix epoch.")
            precise_timestamp = datetime.datetime.fromtimestamp(0)

    return precise_timestamp

def plot_fraction_rings(
    arc_infos: List[pd.DataFrame],
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
        a = infos.iloc[0]['filename-arc']
        arcs.append(a)
        angles = infos[display_angle].tolist()

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
            dirpath = arc_infos[0].iloc[0]['dirpath']
            parts = os.path.normpath(dirpath).split(os.sep)
            title = parts[-2]
    ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()

def plot_tiff_patient_rings(
    info: List[List[pd.DataFrame]],
    n_cols: int = 4,
    **kwargs,
    ) -> None:
    n_fractions = len(info)
    n_rows = int(np.ceil(n_fractions / n_cols))
    _, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': 'polar'}, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, arc_infos in enumerate(info):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        plot_fraction_rings(arc_infos, ax=ax, **kwargs)

    # Hide unused axes.
    for i in range(n_fractions, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_tiff_patient_shrouds(
    shrouds: List[List[Image2D]],
    infos: List[List[List[Dict[str, Any]]]],
    signals: List[List[np.ndarray]] | None = None,
    **kwargs,
    ) -> None:
    for i in range(len(shrouds)):
        signal = signals[i] if signals is not None else None
        plot_fraction_shrouds(shrouds[i], infos[i], signals=signal, **kwargs)

def plot_tiff_breath(
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
    # Plot other frames if provided
    if all_info is not None:
        other_info = all_info[all_info['filepath'] != info['filepath']]
        other_angles = np.deg2rad(other_info['kv-source-angle'].values)
        other_amps = np.array(other_info['MMAmplitude0'].values)
        other_phases = np.array(other_info['MMPhase0'].values)
        # Use a colormap from light to dark for the second colorblind color
        base_color = cb_palette[1]
        n = len(other_angles)
        # Create a light-to-dark gradient for the base color
        colors = [
            tuple(np.clip(np.array(base_color) * (0.5 + 0.5 * (i / max(n-1,1))), 0, 1))
            for i in range(n)
        ]
        # Plot all other frames
        for i in range(n):
            axs[0].scatter(other_angles[i], other_amps[i], color=colors[i], s=30, alpha=0.7, edgecolor='none')
            axs[1].scatter(other_angles[i], other_phases[i], color=colors[i], s=30, alpha=0.7, edgecolor='none')

    # Plot current frame
    angle = np.deg2rad(info['kv-source-angle'])
    amp = info['MMAmplitude0']
    phase = info['MMPhase0']
    axs[0].scatter(angle, amp, color=cb_palette[0], s=60, label='Current frame', zorder=10, edgecolor='black')
    axs[1].scatter(angle, phase, color=cb_palette[0], s=60, label='Current frame', zorder=10, edgecolor='black')

    all_amps = [amp]
    all_phases = [phase]
    if all_info is not None:
        all_amps.extend(other_amps)
        all_phases.extend(other_phases)
    title = f"Breathing amplitude\n[{np.min(all_amps):.2f} - {np.max(all_amps):.2f}mm]"
    axs[0].set_title(title)
    axs[0].set_rlim(min(all_amps), max(all_amps))
    axs[0].set_theta_zero_location('N')
    axs[0].set_theta_direction(-1)
    axs[0].set_yticklabels([])
    title = f"Breathing phase\n[{np.min(all_phases):.2f} - {np.max(all_phases):.2f}°]"
    axs[1].set_title(title, fontsize=title_fontsize)
    axs[1].set_rlim(min(all_phases), max(all_phases))
    axs[1].set_theta_zero_location('N')
    axs[1].set_theta_direction(-1)
    axs[1].set_yticklabels([])

    if show:
        plt.show()

def plot_tiff_projection(
    data: Image2D,
    info: pd.Series,
    ax: mpl.axes.Axes | None = None,
    hist_eq: bool = True,
    normalise: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    window: Window | None = None,
    ) -> None:
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        show = False

    det_size = info['det-size']
    aspect = det_size[1] / det_size[0]
    if normalise:
        data = (data - data.min()) / (data.max() - data.min())
    if hist_eq:
        data = hist_eq_fn(data)

    vmin, vmax = __resolve_window(window, vmin, vmax)

    ax.imshow(data.T, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)
    for s in ax.spines.values():
        s.set_visible(False)
    det_spacing = info['det-spacing']
    ax.set_xlabel(f"LR [{det_spacing[0]}mm]")
    ax.set_ylabel(f"SI [{det_spacing[1]}mm]")

    if show:
        plt.show()

def plot_tiff_image(
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
    window: Window | None = None,
    ) -> None:
    import io
    from PIL import Image as PILImage
    if ax is None:
        fig, ax = plt.subplots()
        show = True
    else:
        fig = None
        show = False

    # Normalise data.
    det_size = info['det-size']
    aspect = det_size[1] / det_size[0]
    if normalise:
        data = (data - data.min()) / (data.max() - data.min())
    if hist_eq:
        data = hist_eq_fn(data)

    # Resolve window to vmin/vmax.
    vmin, vmax = __resolve_window(window, vmin, vmax)

    # Plot slice.
    ax.imshow(data.T, aspect=aspect, cmap='gray', vmin=vmin, vmax=vmax)

    if labels is not None:
        palette = sns.color_palette('colorblind', len(labels))
        for i, l in enumerate(labels):
            cmap_label = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
            ax.imshow(l.T, alpha=alpha_label, cmap=cmap_label)
            ax.contour(l.T, colors=[palette[i]], levels=[.5], linestyles='solid') 

    # Add annotation.
    cdog_version = info['cdog-version']
    title = f"Filename: {os.path.basename(info['filepath'])}\nArc: {info['filename-arc']}, frame: {info['filename-frame']}, angle: {info['filename-angle']}\n\
Imin/max: {data.min():.2f}/{data.max():.2f}, Vmin/max: {vmin}/{vmax}"
    ax.set_title(title, fontsize=title_fontsize)
    # plt.axis('off')
    for s in ax.spines.values():
        s.set_visible(False)
    det_spacing = info['det-spacing']
    ax.set_xlabel(f"LR [{det_spacing[0]}mm]")
    ax.set_ylabel(f"SI [{det_spacing[1]}mm]")

    if show:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return PILImage.open(buf)

def create_tiff_patient_videos(
    pat_path: DirPath,
    fraction: int | List[int] | Literal['all'] = 'all',
    arc: int | List[int] | Literal['all'] = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    fractions = arg_to_list(fraction, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    for f in tqdm(fractions, desc="Creating fractions videos", disable=not show_progress):
        frac_path = os.path.join(pat_path, f"Fx{f:02d}")
        create_tiff_fraction_videos(frac_path, arc=arc, show_progress=show_progress, **kwargs)

def create_tiff_patient_shrouds(
    pat_path: DirPath,
    fraction: int | List[int] | Literal['all'] = 'all',
    arc: int | List[int] | Literal['all'] = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    fractions = arg_to_list(fraction, int, literals={'all': lambda: list_tiff_fractions(pat_path)})
    for f in tqdm(fractions, desc="Creating fraction shrouds", disable=not show_progress):
        frac_path = os.path.join(pat_path, f"Fx{f:02d}")
        create_tiff_fraction_shrouds(frac_path, arc=arc, show_progress=show_progress, **kwargs)

def create_tiff_fraction_shrouds(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    arcs = arg_to_list(arc, int, literals={'all': lambda: list_tiff_arcs(fraction_path)})
    for a in tqdm(arcs, desc=f"Arcs", disable=not show_progress):
        create_tiff_arc_shroud(fraction_path, a, show_progress=show_progress, **kwargs)

def create_tiff_fraction_videos(
    fraction_path: DirPath,
    arc: int | List[int] | Literal['all'] = 'all',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    arcs = arg_to_list(arc, int, literals={'all': lambda: list_tiff_arcs(fraction_path)})
    for a in tqdm(arcs, desc=f"Arcs", disable=not show_progress):
        create_tiff_arc_video(fraction_path, a, show_progress=show_progress, **kwargs)

def create_tiff_arc_video(
    fraction_path: DirPath,
    arc: int,
    hist_eq: bool = True,
    n_frames: int | None = None,
    projections: bool = True,
    proj_method: Literal['binned', 'interp'] = 'interp',
    show_progress: bool = True,
    **kwargs,
    ) -> None:
    data, info = load_tiff_arc(fraction_path, arc, n_frames=n_frames)
    n_frames = data.shape[0]
    if projections:
        filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_video_proj_{proj_method}.apng')
        # Load projection labels.
        set = ds.load('VALKIM-PP')
        pat_i = int(fraction_path.split('\\')[-2].replace('Patient', ''))
        pat_id = f'PAT{pat_i}'
        set = ds.load('VALKIM-PP')
        pat = set.patient(pat_id)
        fraction_str = os.path.basename(fraction_path)
        projpath = os.path.join(set.path, 'data', 'projections', pat.id, fraction_str)
        logger.info(f"Loading projections for arc {arc} from '{projpath}'.")
        ctpath = os.path.join(projpath, f'ct_proj_{proj_method}.npz')
        drrs = load_numpy(ctpath)
        labelpath = os.path.join(projpath, f'labels_proj_{proj_method}.npz')
        labels = load_numpy(labelpath)
    else:
        filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_video.apng')
        labels = None
        drrs = None
    plot_args = [(data[i], info.iloc[i]) for i in range(n_frames)]
    plot_kwargs = [dict(all_info=info, hist_eq=hist_eq, drr=drrs[i] if drrs is not None else None, labels=labels[i] if labels is not None else None, **kwargs) for i in range(n_frames)]
    test = plot_args[0][1]['det-spacing']
    plot_gif(plot_projection, plot_args, plot_kwargs, overwrite=True, savepath=filepath, show=False, show_progress=show_progress, **kwargs)

def plot_tiff_arc_video(
    fraction_path: DirPath,
    arc: int,
    figsize: Tuple[float, float] = (16, 4),
    overwrite: bool = False,
    projections: bool = True,
    **kwargs,
    ) -> None:
    if projections:
        filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_video_proj.apng')
    else:
        filepath = os.path.join(fraction_path, 'kV_proc', f'arc_{arc}_video.apng')
    if overwrite or not os.path.exists(filepath):
        create_tiff_arc_video(fraction_path, arc, figsize=figsize, projections=projections, **kwargs)
    plot_gif(savepath=filepath, **kwargs)

def to_4dct_phase(
    if_phase: float,
    centre_at: float = 0.5,  # Proportion of phase width.
    n_phases: int = 10,
    ) -> int:
    if_phase %= 360
    phase_width = 360 / n_phases
    phase_4dct = (if_phase + (phase_width / 2) - (centre_at * phase_width)) / phase_width
    phase_4dct_int = int(phase_4dct)
    phase_4dct_dec = phase_4dct - phase_4dct_int
    return phase_4dct_int, phase_4dct_dec

def get_kv_path(fraction_path: DirPath) -> DirPath:
    for f in KV_FOLDERS:
        kv_dirpath = os.path.join(fraction_path, f)
        if os.path.isdir(kv_dirpath):
            return kv_dirpath
    raise ValueError(f"No kV subfolder found in '{fraction_path}'. Expected one of: {KV_FOLDERS}")

def assert_arc_jumps(
    fraction_path: DirPath,
    ) -> None:
    # Get processed arc splits.
    kv_dirpath = get_kv_path(fraction_path)
    tiff_files = [f for f in os.listdir(kv_dirpath) if f.endswith('.tiff')]
    arcs = [int(f.split('_')[1]) for f in tiff_files]
    frames = [int(f.split('_')[2]) for f in tiff_files]
    tiff_files, arcs, frames = sort_lists([tiff_files, arcs, frames], key=lambda taf: taf[2])

    # Get non-consecutive frames.
    arcs_diff = np.diff(arcs)
    frames_diff = np.diff(frames)

    # Assert that arcs jump where frames jump.
    jump_idxs = np.argwhere(frames_diff != 1).flatten()

    for i in jump_idxs:
        if arcs_diff[i] == 0:
            raise ValueError(f"Got non-consecutive frames without arc jump at file '{tiff_files[i + 1]}' in '{fraction_path}'.")
