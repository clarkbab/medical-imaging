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
from PIL import Image, TiffTags
from typing import *

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

def get_metadata_props(
    metadata: np.ndarray,
    version: Literal['v2.7', 'v3.0', 'v4.0'],
    base_flag: int = int('0xCAFE', 16),
    ) -> Dict[str, Any]:
    base_pixel = get_base_pixel(metadata, base_flag=base_flag)
    prop_df = load_prop_df(version)
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
            v = load_timestamp(metadata, base_flag=base_flag, base_pixel=base_pixel, prop=p, prop_df=prop_df)
        elif t == 'double':
            v = load_double(metadata, base_flag=base_flag, base_pixel=base_pixel, prop=p, prop_df=prop_df)
        else:
            raise ValueError(f"Unrecognised type '{t}'.")

        # Change units.
        if p == 'PixelHeight' or p == 'PixelWidth':
            v = v * 10  # Convert from cm to mm.
            
        props[p] = v

    return props

def get_prop_offset(
    prop: str,
    prop_df: Optional[pd.DataFrame] = None,
    version: Optional[Literal['v2.7', 'v3.0', 'v4.0']] = None,
    ) -> int:
    if prop_df is None and version is None:
        raise ValueError("Must pass 'prop_df' or 'version' to get offset.")
    if prop_df is None:
        prop_df = load_prop_df(version)
        
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
    img = Image.open(filepath)
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

# Returns image metadata and pixel data.
def load_cdog(
    filepath: str,
    n_image_rows: Optional[int] = None,
    version: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
    # Load image.
    img = Image.open(filepath)
    data = np.array(img).T

    # Get TIFF metadata.
    # We don't really need this, just checks the data type for future calcs.
    # tiff_metadata = load_tiff_metadata(img)

    # Build our own metadata.
    metadata = {}
    metadata['size-x'], metadata['size-y'] = infer_image_size(data, filepath)

    # Extract metadata.
    metadata_image = data[:, metadata['size-y']:]
    metadata['image'] = metadata_image  # For debugging only.
    base_pixel = get_base_pixel(metadata_image)
    if version is not None:
        metadata['version'] = version
    else:
        metadata['version'] = infer_cdog_version(metadata_image)
    props = get_metadata_props(metadata_image, metadata['version'])
    metadata |= props

    # Extract image data.
    image_data = data[:, :metadata['size-y']]

    return metadata, image_data

def load_cdog_df(
    filepath: str,
    ) -> pd.DataFrame:
    m, d = load_cdog(filepath)
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
    version: Literal['v2.7', 'v3.0', 'v4.0'],
    ) -> pd.DataFrame:
    basepath = r"E:\Brett\data\mymi\files\rtf"
    v_str = version.replace('v', '').replace('.', '_')
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

def plot_cdog(
    path: str,
    **kwargs,
    ) -> None:
    if os.path.isdir(path):
        plot_cdog_dirpath(path, **kwargs)
    else:
        plot_cdog_filepath(path, **kwargs)

def plot_cdog_dirpath(
    dirpath: str,
    angle_range: Tuple[Optional[float], Optional[float]] = (None, None),
    arc: Optional[int] = 0,
    max_plots: Optional[int] = 100,
    n_cols: int = 3,
    n_angles: Optional[int] = None,     # Approx. as we'll use consistent spacing to cover the 'angle_range'.
    return_images: bool = False,
    title_fontsize: float = 10,
    **kwargs,
    ) -> Optional[List[PIL.Image]]:
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

    # Remove plots if too many.
    if max_plots is not None and len(tiff_files) > max_plots:
        tiff_files = tiff_files[:max_plots]
        print(f"Truncated to {max_plots} ('max_plots') images.")

    # Full paths.
    tiff_filepaths = [os.path.join(dirpath, f) for f in tiff_files]

    print(f"Plotting {len(tiff_filepaths)} TIFF images.")

    # Plot images.
    if return_images:
        images = []
    else:
        n_images = len(tiff_filepaths)
        n_rows = int(np.ceil(n_images / n_cols))
        _, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), gridspec_kw={ 'hspace': 0.6 }, squeeze=False)

    for i, f in enumerate(tiff_filepaths):
        if return_images:
            image = plot_cdog_filepath(f, return_image=True, title_fontsize=title_fontsize, **kwargs)
            images.append(image)
        else:
            row = i // n_cols
            col = i % n_cols
            plot_cdog_filepath(f, ax=axs[row, col], title_fontsize=title_fontsize, **kwargs)

    if return_images:
        return images

    n_plots = n_rows * n_cols
    n_unused = n_plots - n_images
    for i in range(n_unused):
        axs.flat[-i - 1].set_visible(False)

def plot_cdog_filepath(
    filepath: str,
    ax: Optional[mpl.axes.Axes] = None,
    invert: bool = True,
    return_image: bool = False,
    title_fontsize: float = 10,
    version: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = 1200,
    ) -> Optional[PIL.Image]:
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    filename = os.path.basename(filepath)
    f_arc = '_'.join(filename.split('_')[:2])
    f_frame = filename.split('_')[2]
    f_angle = filename.split('_')[-1].replace('.tiff', '')
    metadata, data = load_cdog(filepath, version=version)
    if version is None:
        version = metadata['version']
    cmap = 'gray_r' if invert else 'gray'
    if vmax is None:
        vmax = np.percentile(data, 99)
    img = ax.imshow(data.T, cmap=cmap, vmin=vmin, vmax=vmax)
    # cbar = fig.colorbar(img)
    title = f"CDOG ({version}) TIFF image ({metadata['size-x']} x {metadata['size-y']})\n\
Arc: {f_arc}, frame: {f_frame}, angle: {f_angle}\n\
Imin/max: {data.min()}/{data.max()}, Vmin/max: {vmin}/{vmax}\n\
MV source angle: {metadata['GantryRtn']:.3f}\n\
kV source/det. angle: {metadata['KVSourceRtn']:.3f}/{metadata['KVDetectorRtn']:.3f}"
    ax.set_title(title, fontsize=title_fontsize)
    # plt.axis('off')
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(f"LR [{metadata['PixelWidth']}mm]")
    ax.set_ylabel(f"SI [{metadata['PixelHeight']}mm]")

    if return_image:
        fig = plt.gcf()
        buffer = io.BytesIO()
        fig.savefig(buffer, bbox_inches='tight', dpi=100, format='png')
        plt.close(fig)
        buffer.seek(0)
        return np.asarray(Image.open(buffer))

    if show:
        plt.show()

def plot_cdog_gif(
    dirpath: str,
    *args,
    arc: int = 0,
    end_time: float = 5,
    frame_time: float = 0.5,
    loop: bool = True,
    overwrite: bool = False,
    width: float = 500,
    **kwargs,
    ) -> None:
    # Get filepath.
    files = os.listdir(dirpath)
    tiff_files = list(sorted([f for f in files if f.endswith('.tiff')]))
    arcs = list([str(a) for a in np.unique(['_'.join(f.split('_')[:2]) for f in tiff_files])])
    arc_str = arcs[arc]
    filepath = os.path.join(dirpath, f"{arc_str}.gif")

    if overwrite or not os.path.exists(filepath):
        # Get tiff images.
        png_images = plot_cdog_dirpath(dirpath, *args, arc=arc, return_images=True, **kwargs)

        # Save gif.
        frames = png_images
        frames_per_second = 1 / frame_time
        frames = frames + [frames[-1]] * int(end_time / frame_time)
        imageio.mimsave(filepath, frames, fps=frames_per_second, loop=0 if loop else None)

    display(IPythonImage(filename=filepath, width=500))

def plot_cdog_hist(
    filepath: str,
    **kwargs,
    ) -> None:
    _, data = load_cdog(filepath)
    plt.hist(data)
    plt.show()
