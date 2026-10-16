# Marker Editor — Streamlit app
#
# Loads a marker CSV (columns: Marker_0_X, Marker_0_Y, Marker_1_X, …) and a
# NumPy projection image stack (.npz, first axis = frames).  Displays each frame
# with markers overlaid on an interactive canvas.  Drag yellow dots to reposition
# markers; click Save to write the updated CSV (to a clone by default).
#
# Image processing matches plot_projection: invert intensities → histogram-equalize.
#
# Coordinate notes
# ─────────────────
# The kV .npz files produced by process_tiff_arc() are stored transposed:
#   shape = (n_frames, W, H)   [first pixel axis = detector LR / width]
# Enable "Transpose image" (default on) to display them as (H, W) images.
# The marker CSV uses a bottom-left origin for Y; enable "Flip Y" (default on)
# to convert to the top-left canvas origin used by the browser.

import base64
import io
import json
import shutil
import tkinter as tk
import traceback
from tkinter import filedialog
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc
from PIL import Image as PILImage
from skimage.exposure import equalize_hist

# ── Native file-browse helper ─────────────────────────────────────────────────
def _initial_dir(state_key: str) -> str | None:
    """Return the parent directory of the current value of session_state[state_key], if it exists."""
    val = st.session_state.get(state_key, '')
    if val:
        p = Path(val)
        d = p.parent if p.is_file() else p
        if d.exists():
            return str(d)
    return None

# Browse helpers write to a '_pending_<key>' slot so that the actual widget key
# is updated *before* the widget is instantiated on the next rerun (Streamlit
# raises an error if you set a widget key after the widget has already run).
def _browse_file(state_key: str, filetypes: list, title: str = 'Select file') -> None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes,
                                      initialdir=_initial_dir(state_key), parent=root)
    root.destroy()
    if path:
        st.session_state[f'_pending_{state_key}'] = path

def _browse_save(state_key: str, filetypes: list, title: str = 'Save as') -> None:
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    path = filedialog.asksaveasfilename(title=title, filetypes=filetypes,
                                        initialdir=_initial_dir(state_key),
                                        confirmoverwrite=False, parent=root)
    root.destroy()
    if path:
        st.session_state[f'_pending_{state_key}'] = path

# ── Session persistence ──────────────────────────────────────────────────────
_STATE_FILE = Path.home() / '.marker_editor_state.json'
_PERSISTED_KEYS = ('csv_path', 'npz_path', 'marker_style', 'marker_thickness', 'marker_opacity', 'marker_mode', 'arrow_step')

def _load_persisted_state() -> None:
    """Restore saved widget values on first run of a new session."""
    try:
        data = json.loads(_STATE_FILE.read_text())
    except Exception:
        data = {}
    for k in _PERSISTED_KEYS:
        if k not in st.session_state and k in data:
            st.session_state[k] = data[k]

def _save_persisted_state() -> None:
    """Write current widget values to disk."""
    data = {k: st.session_state.get(k, '') for k in _PERSISTED_KEYS}
    try:
        _STATE_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass

# ── Custom component ──────────────────────────────────────────────────────────
import mymi as _mymi
_COMP_DIR = Path(_mymi.__file__).parent / 'apps' / 'static' / 'marker_drag'
_marker_drag = stc.declare_component('marker_drag', path=str(_COMP_DIR))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(layout='wide', page_title='Marker Editor')
st.title('Marker Editor')

# ── Session state init ────────────────────────────────────────────────────────
_ss = st.session_state
_load_persisted_state()  # restore paths from last session before any widgets run
_defaults = {
    'loaded':          False,
    'images':          None,   # original ndarray, shape (n_frames, dim0, dim1)
    'markers':         None,   # (n_frames, n_markers, 2) float — CSV space [x, y]
    'orig_markers':    None,   # immutable copy from CSV for Reset
    'orig_df':         None,
    'marker_xy_cols':  [],     # [(x_col, y_col), …]
    '_out_path':       '',
    'n_frames':        0,
    'undo_stack':      [],     # list of {'frame': int, 'positions': ndarray}
    '_last_drag_sig':  None,
    '_crop_cache':     {},     # frame_idx -> (cx0, cy0, cx1, cy1), frozen while nudging with arrow keys
}
for k, v in _defaults.items():
    if k not in _ss:
        _ss[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
# ── Apply any pending browse results before widgets are instantiated ────────
for _pk in ('csv_path', 'npz_path'):
    _pend = f'_pending_{_pk}'
    if _pend in _ss:
        _ss[_pk] = _ss.pop(_pend)

# Apply any pending frame navigation (Prev/Next) before the sidebar runs, so that
# sidebar widgets reading 'frame_slider' (e.g. the Copy-from-previous-frame button's
# enabled state) see the up-to-date frame instead of lagging by one click.
if '_pending_frame' in _ss:
    _ss['frame_slider'] = _ss.pop('_pending_frame')

with st.sidebar:
    st.header('Files')

    _CSV_DEFAULT = (
        r'R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\MilesManualMarkerSegmentation'
        r'\Patient01\Fx01\Arc01\MarkerLocationsGA_CouchShift_adapted.txt'
    )
    _NPZ_DEFAULT = (
        r'R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files'
        r'\Patient01\Fx01\kV_proc\arc_1.npz'
    )
    for _k, _v in [('csv_path', _CSV_DEFAULT), ('npz_path', _NPZ_DEFAULT)]:
        if _k not in _ss:
            _ss[_k] = _v

    # Marker CSV
    _c1, _c2 = st.columns([4, 1])
    with _c1:
        csv_path = st.text_input('Marker CSV', key='csv_path')
    with _c2:
        st.write('')  # vertical nudge
        if st.button('📂', key='browse_csv', help='Browse for marker CSV'):
            _browse_file('csv_path',
                         [('Text / CSV', '*.txt *.csv'), ('All files', '*.*')],
                         'Select marker CSV')
            st.rerun()

    # NPZ images
    _c1, _c2 = st.columns([4, 1])
    with _c1:
        npz_path = st.text_input('NPZ images', key='npz_path')
    with _c2:
        st.write('')
        if st.button('📂', key='browse_npz', help='Browse for NPZ image stack'):
            _browse_file('npz_path',
                         [('NumPy archive', '*.npz'), ('All files', '*.*')],
                         'Select NPZ image stack')
            st.rerun()

    st.header('Display')
    transpose_img = st.checkbox(
        'Transpose image (W,H → H,W)',
        value=True,
        help='Enable if the image appears rotated. The kV .npz files are stored '
             'as (n_frames, W, H) and need transposing for correct display.',
    )
    flip_y = st.checkbox(
        'Flip Y markers',
        value=True,
        help='Convert marker Y from bottom-left origin (CSV convention) to '
             'top-left origin (canvas convention).',
    )
    flip_v = st.checkbox(
        'Flip image vertically',
        value=True,
        help='Flip the display image (and markers) upside-down. '
             'Marker positions are automatically unflipped before saving.',
    )
    zoom_disabled = st.checkbox(
        'Disable zoom',
        value=False,
        help='Show the full image instead of zooming in to the marker bounding box.',
    )
    zoom_border = st.number_input(
        'Zoom border (px)', min_value=0, value=100, step=10,
        help='Padding (in image pixels) around the marker bounding box in the zoomed view.',
        disabled=zoom_disabled,
    )

    st.header('Markers')
    for _k, _v in [('marker_style', 'cross'), ('marker_thickness', 2.5), ('marker_opacity', 0.5), ('marker_mode', 'arrow'), ('arrow_step', 0.5)]:
        if _k not in _ss:
            _ss[_k] = _v
    marker_mode = st.radio(
        'Marker move mode', ['arrow', 'drag', 'swap'], key='marker_mode',
        format_func=lambda v: {
            'arrow': 'Click to select, arrow keys to move',
            'drag': 'Click and drag',
            'swap': 'Click a marker, type another index to swap',
        }[v],
        help='Mode 1 (default): click a marker to select it, then nudge it with arrow keys. '
             'Mode 2: the original click-and-drag behaviour. '
             'Mode 3: click a marker, then type another marker\'s index (e.g. 0, 1, 2) and press '
             'Enter in the popup to swap their positions.',
    )
    arrow_step = st.number_input(
        'Arrow key step (px)', min_value=0.1, step=0.1, key='arrow_step',
        help='Distance (in image pixels) moved per arrow-key press. Only used in arrow-key mode.',
    )
    marker_style = st.selectbox(
        'Marker style', ['cross', 'point'], key='marker_style',
        help='Shape used to display markers on the canvas.',
    )
    marker_thickness = st.slider(
        'Marker thickness', min_value=1.0, max_value=8.0, step=0.5, key='marker_thickness',
        help='Line width (cross) or radius (point) of the displayed markers.',
    )
    marker_opacity = st.slider(
        'Marker opacity', min_value=0.1, max_value=1.0, step=0.05, key='marker_opacity',
        help='Opacity of the displayed markers.',
    )
    _save_persisted_state()  # Persist marker display settings as soon as they change, not just on Load.

    st.divider()
    # frame_slider isn't defined yet this run (the slider widget runs later, in the
    # main body) - fall back to the persisted value from the previous run.
    _cur_frame = _ss.get('frame_slider', 0)
    _can_copy_prev = _ss.loaded and _cur_frame > 0
    if st.button('Copy from previous frame', key='btn_copy_prev',
                 disabled=not _can_copy_prev, use_container_width=True,
                 help='Overwrite the current frame\'s marker positions with those from the '
                      'previous frame.'):
        _ss.undo_stack.append({'frame': _cur_frame, 'positions': _ss.markers[_cur_frame].copy()})
        if len(_ss.undo_stack) > 50:
            _ss.undo_stack.pop(0)
        _ss.markers[_cur_frame] = _ss.markers[_cur_frame - 1].copy()
        _ss['_crop_cache'].pop(_cur_frame, None)  # recenter crop after a larger, non-nudge change
        st.rerun()

    st.divider()
    load_btn = st.button('Load files', type='primary')

# Inject JS so sidebar text inputs show their full value as a native browser
# tooltip when the user hovers directly over the input field.
# Uses polling (not just an 'input' listener) because the Browse-dialog flow updates
# the value programmatically via session_state + st.rerun(), which doesn't fire a
# native 'input' DOM event - so a listener-only patch left the tooltip stale.
stc.html("""
<script>
(function () {
  function patch() {
    var inputs = window.parent.document.querySelectorAll(
      '[data-testid="stSidebar"] [data-testid="stTextInput"] input'
    );
    inputs.forEach(function (inp) {
      if (inp.title !== inp.value) {
        inp.title = inp.value || '';
      }
    });
  }
  setInterval(patch, 300);
})();
</script>
""", height=1)

if load_btn:
    _bar = st.progress(0, text='Reading CSV …')
    try:
        # ── CSV ───────────────────────────────────────────────────────────
        df = pd.read_csv(csv_path)
        x_cols = sorted([c for c in df.columns if c.startswith('Marker_') and c.endswith('_X')])
        y_cols = [c.replace('_X', '_Y') for c in x_cols]
        y_cols = [c for c in y_cols if c in df.columns]
        x_cols = x_cols[:len(y_cols)]
        if not x_cols:
            _bar.empty()
            st.sidebar.error('No Marker_N_X / Marker_N_Y columns found in CSV.')
            st.stop()

        xs = df[x_cols].to_numpy(dtype=float)   # (n_frames, n_markers)
        ys = df[y_cols].to_numpy(dtype=float)
        markers = np.stack([xs, ys], axis=-1)    # (n_frames, n_markers, 2)
        _bar.progress(33, text='Loading images …')

        # ── NPZ ──────────────────────────────────────────────────────────
        npz = np.load(npz_path)
        raw_images = npz[npz.files[0]] if hasattr(npz, 'files') else npz
        raw_images = raw_images.astype(np.float32)

        # Auto-crop kV detector metadata band using companion info CSV if present.
        # NPZ shape: (n_frames, W, H+meta).  det-size = (W, H) → crop axis 2 to H.
        import ast as _ast
        _info_csv_auto = Path(npz_path).parent / (Path(npz_path).stem + '_info.csv')
        if _info_csv_auto.exists():
            try:
                _det_size = _ast.literal_eval(
                    str(pd.read_csv(_info_csv_auto, nrows=1)['det-size'].iloc[0])
                )
                if raw_images.shape[2] > _det_size[1]:
                    raw_images = raw_images[:, :, :_det_size[1]]
            except Exception:
                pass  # info CSV malformed – use full image

        # Reconcile CSV row count with image frame count.
        n_img_frames = raw_images.shape[0]
        n_csv_frames = len(df)
        if n_csv_frames < n_img_frames:
            n_pad = n_img_frames - n_csv_frames
            pad_df = pd.DataFrame({c: [np.nan] * n_pad for c in df.columns})
            df = pd.concat([df, pad_df], ignore_index=True)
            pad_markers = np.full((n_pad, markers.shape[1], 2), np.nan, dtype=float)
            markers = np.concatenate([markers, pad_markers], axis=0)
        elif n_csv_frames > n_img_frames:
            df = df.iloc[:n_img_frames].reset_index(drop=True)
            markers = markers[:n_img_frames]

        _bar.progress(66, text='Preparing output …')

        # ── Output path (always <base>_edited<ext>, never _edited_edited) ──────
        _p = Path(csv_path)
        import re as _re
        _base = _re.sub(r'(_edited)+$', '', _p.stem)
        out_path = str(_p.with_stem(_base + '_edited'))
        if not Path(out_path).exists():
            shutil.copy2(csv_path, out_path)

        _ss.images            = raw_images
        _ss.markers           = markers
        _ss.orig_markers      = markers.copy()
        _ss.orig_df           = df
        _ss.marker_xy_cols    = list(zip(x_cols, y_cols))
        _ss['_out_path']      = out_path
        _ss.n_frames          = len(df)
        _ss.undo_stack        = []
        _ss['_last_drag_sig'] = None
        _ss['_crop_cache']    = {}
        _ss.loaded            = True

        _bar.progress(100, text='Done.')
        _bar.empty()
        _save_persisted_state()
        n_m = len(x_cols)
        st.sidebar.success(
            f'Loaded {_ss.n_frames} frames, {n_m} marker{"s" if n_m != 1 else ""}.\n\n'
            f'CSV: `{csv_path}`\n\n'
            f'NPZ: `{npz_path}`\n\n'
            f'Output: `{out_path}`'
        )
    except Exception:
        _bar.empty()
        st.sidebar.error(f'Load failed:\n```\n{traceback.format_exc()}\n```')
        st.stop()

# ── Guard ─────────────────────────────────────────────────────────────────────
if not _ss.loaded:
    st.info('Use the sidebar to load a marker CSV and an NPZ image stack.')
    st.stop()

# ── Per-render helpers ────────────────────────────────────────────────────────
images  = _ss.images   # (n_frames, dim0, dim1)
markers = _ss.markers  # (n_frames, n_markers, 2)

def _effective_frame(frame_idx: int, do_transpose: bool):
    """Return (float64 H×W array, H, W) for the given frame."""
    raw = images[frame_idx]
    if do_transpose:
        raw = raw.T
    H, W = raw.shape
    return raw.astype(np.float64), H, W

def _process_image(data: np.ndarray) -> np.ndarray:
    """Invert + histogram-equalise → uint8 H×W."""
    inverted = -data
    eq       = equalize_hist(inverted)          # → [0, 1] float
    return (eq * 255).astype(np.uint8)

def _to_data_url(uint8_hw: np.ndarray) -> str:
    pil = PILImage.fromarray(uint8_hw, mode='L').convert('RGB')
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=85)    # JPEG is faster to transmit than PNG
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

def _markers_to_canvas(frame_markers: np.ndarray, H: int, do_flip_y: bool, do_flip_v: bool = False):
    """Convert (n_markers, 2) CSV [x, y] → list of canvas-space dicts."""
    out = []
    for i, (x, y) in enumerate(frame_markers):
        cy = (H - y) if do_flip_y else y
        if do_flip_v:
            cy = H - 1 - cy
        out.append({'label': str(i), 'x': float(x), 'y': float(cy)})
    return out

def _canvas_to_csv(cx: float, cy: float, H: int, do_flip_y: bool, do_flip_v: bool = False):
    """Convert canvas-space (cx, cy) → CSV-space (x, y)."""
    x = cx
    if do_flip_v:
        cy = H - 1 - cy
    y = (H - cy) if do_flip_y else cy
    return x, y

# ── Frame slider + navigation ─────────────────────────────────────────────────────
# (any pending navigation from Prev/Next is already resolved before the sidebar runs - see above)
frame_idx = st.slider(
    'Frame',
    min_value=0,
    max_value=_ss.n_frames - 1,
    value=0,
    key='frame_slider',
)

_has_undo = len(_ss.undo_stack) > 0
_undo_lbl = '↩ Undo' + (f' (Fr. {_ss.undo_stack[-1]["frame"] + 1})' if _has_undo else '')
_nc1, _nc2, _nc3, _nc4, _ = st.columns([1, 1, 1.5, 1, 5])
with _nc1:
    if st.button('◄ Prev', disabled=frame_idx == 0,
                 key='btn_prev', use_container_width=True):
        _ss['_pending_frame'] = frame_idx - 1
        st.rerun()
with _nc2:
    if st.button('Next ►', disabled=frame_idx == _ss.n_frames - 1,
                 key='btn_next', use_container_width=True):
        _ss['_pending_frame'] = frame_idx + 1
        st.rerun()
with _nc3:
    if st.button(_undo_lbl, disabled=not _has_undo,
                 key='btn_undo', use_container_width=True):
        _entry = _ss.undo_stack.pop()
        _ss.markers[_entry['frame']] = _entry['positions']
        _ss['_crop_cache'].pop(_entry['frame'], None)  # recenter crop after a larger, non-nudge change
        st.rerun()
with _nc4:
    if st.button('↺ Reset', key='btn_reset', use_container_width=True):
        _ss.markers[frame_idx] = _ss.orig_markers[frame_idx].copy()
        _ss.undo_stack = [e for e in _ss.undo_stack if e['frame'] != frame_idx]
        _ss['_crop_cache'].pop(frame_idx, None)  # recenter crop after a larger, non-nudge change
        st.rerun()

# ── Build image for current frame ─────────────────────────────────────────────
frame_data, H, W = _effective_frame(frame_idx, transpose_img)
img_uint8         = _process_image(frame_data)
if flip_v:
    img_uint8 = np.flipud(img_uint8)
img_src           = _to_data_url(img_uint8)

# ── Canvas dimensions ─────────────────────────────────────────────────────────
MAX_CANVAS_W = 580   # fits inside one of two equal columns in wide layout
scale         = MAX_CANVAS_W / W
canvas_w      = MAX_CANVAS_W
canvas_h      = max(1, int(H * scale))

frame_markers = markers[frame_idx]  # (n_markers, 2)
marker_list   = _markers_to_canvas(frame_markers, H, flip_y, flip_v)

# ── Crop to marker bounding box + border ────────────────────────────────────────
# The crop follows the current marker positions in 'drag' (and 'swap') mode, so the
# view zooms/pans as markers move. In 'arrow' mode the crop is frozen (cached per
# frame) so small keyboard nudges don't shift the view - it's recomputed only the
# first time a frame is seen, or whenever a drag move updates the cache.
# Each component result reports the crop origin it was rendered with (crop_x0/
# crop_y0 below) so positions are always converted using the correct origin, even
# if the crop shifts between renders - that mismatch was the original cause of
# other markers appearing to jump.
if zoom_disabled:
    _cx0, _cy0, _cx1, _cy1 = 0, 0, W, H
elif marker_mode == 'arrow' and frame_idx in _ss['_crop_cache']:
    _cx0, _cy0, _cx1, _cy1 = _ss['_crop_cache'][frame_idx]
else:
    _vx = [m['x'] for m in marker_list if not (np.isnan(m['x']) or np.isnan(m['y']))]
    _vy = [m['y'] for m in marker_list if not (np.isnan(m['x']) or np.isnan(m['y']))]
    if _vx and _vy:
        _cx0 = max(0,   int(min(_vx)) - zoom_border)
        _cx1 = min(W,   int(max(_vx)) + zoom_border + 1)
        _cy0 = max(0,   int(min(_vy)) - zoom_border)
        _cy1 = min(H,   int(max(_vy)) + zoom_border + 1)
    else:
        _cx0, _cy0, _cx1, _cy1 = 0, 0, W, H
    _ss['_crop_cache'][frame_idx] = (_cx0, _cy0, _cx1, _cy1)

img_crop   = img_uint8[_cy0:_cy1, _cx0:_cx1]
_crop_h, _crop_w = img_crop.shape
img_src    = _to_data_url(img_crop)
marker_list = [
    {'label': m['label'], 'x': m['x'] - _cx0, 'y': m['y'] - _cy0}
    for m in marker_list
]

_col_l, _col_r = st.columns(2)
with _col_l:
    result = _marker_drag(
        img_src   = img_src,
        markers   = marker_list,
        img_w     = float(_crop_w),
        img_h     = float(_crop_h),
        canvas_w  = canvas_w,
        canvas_h  = canvas_h,
        frame_idx = frame_idx,
        crop_x0   = float(_cx0),
        crop_y0   = float(_cy0),
        marker_style      = marker_style,
        marker_thickness = float(marker_thickness),
        marker_opacity   = float(marker_opacity),
        marker_mode      = marker_mode,
        arrow_step       = float(arrow_step),
        key       = 'marker_drag',
        default   = None,
    )
with _col_r:
    st.image(img_crop, use_container_width=True, caption='Reference (no markers)')

# ── Apply drag result ─────────────────────────────────────────────────────────
def _drag_sig(r):
    if r is None: return None
    return (r.get('frame_idx'), tuple((m['x'], m['y']) for m in r.get('markers', [])))

if result is not None and result.get('frame_idx') == frame_idx:
    _sig = _drag_sig(result)
    if _sig != _ss.get('_last_drag_sig'):
        _ss.undo_stack.append({'frame': frame_idx, 'positions': _ss.markers[frame_idx].copy()})
        if len(_ss.undo_stack) > 50:
            _ss.undo_stack.pop(0)
        _ss['_last_drag_sig'] = _sig
        # Use the crop origin the component actually rendered with, not the current one.
        _rx0 = float(result.get('crop_x0', _cx0))
        _ry0 = float(result.get('crop_y0', _cy0))
        for i, m in enumerate(result.get('markers', [])):
            if i >= markers.shape[1]:
                break
            x_csv, y_csv = _canvas_to_csv(
                float(m['x']) + _rx0, float(m['y']) + _ry0, H, flip_y, flip_v
            )
            _ss.markers[frame_idx, i, 0] = x_csv
            _ss.markers[frame_idx, i, 1] = y_csv
        # Rerun so marker_list is recomputed from the updated positions before
        # the component receives its next args (without this the update is one
        # interaction behind because marker_list was built before result arrived).
        st.rerun()

# ── Bottom controls ───────────────────────────────────────────────────────────
st.divider()
col_tbl, col_save, _ = st.columns([2, 1, 3])

with col_tbl:
    n_m = markers.shape[1]
    st.dataframe(
        pd.DataFrame(
            {
                'X': [f"{markers[frame_idx, i, 0]:.1f}" for i in range(n_m)],
                'Y': [f"{markers[frame_idx, i, 1]:.1f}" for i in range(n_m)],
            },
            index=[f'Marker {i}' for i in range(n_m)],
        ),
        use_container_width=True,
        key=f'marker_pos_{frame_idx}',
    )

with col_save:
    st.write('')  # vertical alignment
    st.write('')
    if st.button('Save to File', type='primary', use_container_width=True):
        df_out = _ss.orig_df.copy()
        for fi in range(_ss.n_frames):
            for mi, (x_col, y_col) in enumerate(_ss.marker_xy_cols):
                df_out.at[fi, x_col] = _ss.markers[fi, mi, 0]
                df_out.at[fi, y_col] = _ss.markers[fi, mi, 1]
        df_out.to_csv(_ss['_out_path'], index=False)
        st.success(f'Saved to  \n`{_ss["_out_path"]}`')
