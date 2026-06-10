# Tenets (for Claude and us):
#
# 1. Contour offsets, confidence and clinical flags are all saved to "contour_shifts.csv"
#    and are persisted every time we change projection image. These values are also loaded
#    if the file already exists, and new values are merged with the old. We use merge
#    because we might be loading a subset (e.g. n=10) of projections, and we don't want to
#    lose the offsets for the other projections.
# 2. There are four ways to change the contour offset: 1) drag the contour, 2) move the
#    x/y offset sliders (either drag them, or use the left/right arrow buttons), 3) reset
#    the offset to (0, 0), 4) an undo button that returns the contour to it's previous
#    position for any string of changes. Any of these interactions should change the session
#    state of the offset, and then all of these components should be updated so that the UI 
#    reflects the current offset.
# 3. Coordinate systems. Some .his/.xim files have gantry angles (and maybe other values)
#    encoded in different formats from IEC61217 (the system this app uses). We offer a way
#    to convert from other systems in the UI.
# 4. Loading from CT dicom is the slowest component in the pipeline and can be easily cached.
#    We do this by computing a hash of the file names and modification times and storing the
#    CT data in 'ct_<hash>.npz'. If the hash changes, we recomputed the cached CT image.
# 5. Hotkeys: Left/right/up/down arrow keys move the contour when in contour edit mode. To
#    move to the next projection image, we can use Shift + left/right arrow keys. To toggle
#    between different window modes, we can use M/m (Contour move), Z/z (Zoom), D/d (Deform).
# 6. Some user settings should be persisted between sessions, see example_session.json.
#    Some settings depend on other settings, e.g. we should remember the patient, but only if
#    the site folder is the same as the last session. If the site folder changes, we should
#    clear out all dependent settings recursively. A copy of the session state should be kept
#    in memory during the session and updated immediately when field changes are made (either
#    by the user, or by the code (auto_select=True)). When the file is persisted it should
#    be a direct copy of the current session state.
# 7. Users don't know about 0-based indexing. Store all state using 0-based indexing, but
#    display using 1-based indexing.
# 8. After finalise planning has been clicked, the selected fraction in treatment images should
#    be synced with the selected fraction in planning data - if present, planning data doesn't
#    typically have fractions.
# 9. Accepted projection file types are .his, .xim, and .tiff.
# 10. Auto-select is used to make the user experience easier. For example, when a site folder
#    is chosen, the list of patients is found at the expected path if possible. Then the first
#    patient is auto-selected. This contrasts with manual selection, where the patient is 
#    chosen by the user from the drop-down. This feature is more important for other fields,
#    like CT, RTPLAN, RTSTRUCT files, which can be auto-selected to save the user a lot of time.
# 11. For .his files, if Frames.xml is missing log "Detector offsets couldn't be loaded, missing
#    file: <filepath>". If Frames.xml is present but missing the "Frames" element, log "Detector
#    offsets couldn't be loaded, missing 'Frames' element in: <filepath>".
# 12. There should be a clear separation between session state and UI. For example, if the fraction
#    drop-down changes, this should immediately change "fraction" in the session state. Then, 
#    any other components that depend on "fraction" should update based on the new value in 
#    the session state. E.g. auto-detect paths should be searched again using a new base path
#    using the new fraction from the session state.
# 13. The "prospective" slider switches between prospective and retrospective modes. In retrospective
#     mode (the default), we have access to treatment images and these are displayed alongside 
#     the projections. In prospective mode, we don't have treatment images, and we just display 
#     projections. Certain geometry that is loaded from treatment images in retro mode must be
#     added manually in prospective mode. For example, the machine, number of projections (default=35)
#     and start/stop angles. 

import base64
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pydicom as dcm
import streamlit as st
import streamlit.components.v1 as stc
import torch
import traceback

from dicomset.dicom.utils import from_ct_dicom, from_rtstruct_dicom, list_rtstruct_regions
from dicomset.utils import affine_spacing, hist_eq, save_numpy
from mymi.utils.cdog import load_tiff
from mymi.utils.elekta import HisFrameInfo, load_his, load_his_angles_and_files
from mymi.utils.varian import load_xim, load_xim_angles_and_files
from mymi.utils.projections import project_diffdrr, project_ctorch, project_igt

_STATE_FILE = Path.home() / '.contour_alignment_state.json'

def _load_persisted_state() -> None:
    try:
        data = json.loads(_STATE_FILE.read_text())
        st.session_state['_state_file_loaded'] = str(_STATE_FILE)
    except Exception:
        data = {}

    # _persisted is the in-memory mirror of the JSON — initialise once per session.
    if '_persisted' not in st.session_state:
        st.session_state['_persisted'] = data

    # Restore global settings.
    if 'method' not in st.session_state:
        st.session_state['method'] = data.get('projection_method', 'CTorch')
    img = data.get('image_settings', {})
    for _jkey, _skey, _default in [
        ('hist_eq', 'use_hist_eq', True),
        ('zoom_step', 'zoom_step', 10),
        ('contour_visible', 'contour_visible', True),
        ('contour_linestyle', 'contour_linestyle', 'Solid'),
        ('contour_color', 'contour_color', '#ff3c3c'),
        ('show_margin', 'show_margin', True),
        ('margin_width_mm', 'margin_width_mm', 10.0),
        ('onscreen_threshold', 'onscreen_threshold', 50.0),
    ]:
        if _skey not in st.session_state:
            st.session_state[_skey] = img.get(_jkey, _default)

    # Restore widget session-state keys from the persisted data.
    def _restore(json_key: str, session_key: str) -> None:
        name = data.get(json_key, {}).get('name', '')
        if session_key not in st.session_state and name:
            st.session_state[session_key] = name

    _site_name = data.get('site', {}).get('name', '')
    if 'data_dir' not in st.session_state and _site_name:
        st.session_state['data_dir'] = _site_name
        st.session_state['_data_dir_source'] = 'session'

    _restore('patient',          'patient')
    _restore('ct',               'ct_dir')
    _restore('rtplan',           'plan_path')
    _restore('rtstruct',         'rtstruct_path')
    _restore('region',           'selected_region')
    _restore('imaging_type',     'treat_imaging_type')
    _restore('session',          'cbct_session')
    _restore('treatment_images', 'treatment_images')
    # Restore prospective — boolean, must not auto-restore planning_finalised.
    _prosp_name = data.get('prospective', {}).get('name', False)
    if 'prospective_mode' not in st.session_state and _prosp_name is True:
        st.session_state['prospective_mode'] = True

    # fraction restores both planning and treatment fraction widget keys.
    _frac_name = data.get('fraction', {}).get('name', '')
    if 'fraction' not in st.session_state and _frac_name:
        st.session_state['fraction'] = _frac_name
        st.session_state['treat_fraction'] = _frac_name


def _save_persisted_state(log_key: str = '') -> None:
    try:
        p = st.session_state.get('_persisted', {})
        # Refresh globals — these have no auto_select tracking.
        p['projection_method'] = st.session_state.get('method', 'CTorch')
        p['image_settings'] = {
            'hist_eq':            st.session_state.get('use_hist_eq', True),
            'zoom_step':          st.session_state.get('zoom_step', 10),
            'contour_visible':    st.session_state.get('contour_visible', True),
            'contour_linestyle':  st.session_state.get('contour_linestyle', 'Solid'),
            'contour_color':      st.session_state.get('contour_color', '#ff3c3c'),
            'show_margin':        st.session_state.get('show_margin', True),
            'margin_width_mm':    st.session_state.get('margin_width_mm', 10.0),
            'onscreen_threshold': st.session_state.get('onscreen_threshold', 50.0),
        }
        # site is always user-set; refresh name from session state.
        p['site'] = {'name': st.session_state.get('data_dir', ''), 'auto_select': False}
        st.session_state['_persisted'] = p
        _STATE_FILE.write_text(json.dumps(p, indent=2))
        if log_key:
            st.session_state[log_key] = str(_STATE_FILE)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Serve component HTML via Streamlit's own static file server (same port,
# no firewall issues).  Requires --server.enableStaticServing=true or:
#   [server]
#   enableStaticServing = true
# in .streamlit/config.toml next to this script.
# ---------------------------------------------------------------------------
_APP_DIR = Path(__file__).parent
_STATIC_DIR = _APP_DIR / 'static'

_DRAG_STATIC_DIR = _STATIC_DIR / 'contour_drag'

_contour_drag = stc.declare_component('contour_drag', path=str(_DRAG_STATIC_DIR))


# ---------------------------------------------------------------------------
# Confidence panel component — injects a fixed-position overlay into the
# parent page so it stays visible as the user scrolls.
# ---------------------------------------------------------------------------
_CONF_STATIC_DIR = _STATIC_DIR / 'confidence_panel'

_confidence_panel = stc.declare_component('confidence_panel', path=str(_CONF_STATIC_DIR))


def _img_to_src(arr: np.ndarray, vmin: float, vmax: float) -> str:
    """Encode a 2-D float array as a base64 grayscale PNG data-URI."""
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    try:
        from PIL import Image as _PilImage
        _PilImage.fromarray(norm, 'L').save(buf, format='PNG')
    except ImportError:
        plt.imsave(buf, norm, cmap='gray', format='png')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


_PATH_SOURCE_KEYS = ('ct_dir', 'rtstruct_path', 'plan_path', 'treatment_images')

# Maps session-state widget key → JSON persistence key.
_SESSION_TO_JSON_KEY: dict[str, str] = {
    'patient':            'patient',
    'ct_dir':             'ct',
    'plan_path':          'rtplan',
    'rtstruct_path':      'rtstruct',
    'selected_region':    'region',
    'treat_fraction':     'fraction',
    'fraction':           'fraction',
    'treat_imaging_type': 'imaging_type',
    'cbct_session':       'session',
    'prospective_mode':   'prospective',
    'treatment_images':   'treatment_images',
}

# Maps JSON key → its parent JSON key (written as depends_on when auto_select).
_JSON_PARENT: dict[str, str] = {
    'patient':          'site',
    'fraction':         'patient',
    'ct':               'patient',
    'rtplan':           'patient',
    'rtstruct':         'patient',
    'region':           'rtstruct',
    'imaging_type':     'fraction',
    'session':          'imaging_type',
    'treatment_images': 'session',
    'prospective':      'patient',
}

_DETECTOR_SIZES: dict[str, tuple[int, int]] = {
    '512x512':   (512,  512),
    '1024x768':  (1024, 768),
    '1024x1024': (1024, 1024),
}


def _detector_size_label(w: int, h: int) -> str:
    for label, (lw, lh) in _DETECTOR_SIZES.items():
        if w == lw and h == lh:
            return label
    return '1024x768'


def _mark_path_manual(session_key: str) -> None:
    jk = _SESSION_TO_JSON_KEY.get(session_key)
    if not jk:
        return
    p = st.session_state.get('_persisted', {})
    p[jk] = {'name': st.session_state.get(session_key, ''), 'auto_select': False}
    st.session_state['_persisted'] = p


def _mark_path_auto(session_key: str) -> None:
    jk = _SESSION_TO_JSON_KEY.get(session_key)
    if not jk:
        return
    p = st.session_state.get('_persisted', {})
    name = st.session_state.get(session_key, '')
    entry: dict = {'name': name, 'auto_select': True}
    parent = _JSON_PARENT.get(jk)
    if parent and name:
        entry['depends_on'] = parent
    p[jk] = entry
    st.session_state['_persisted'] = p


def _clear_persisted(*session_keys: str) -> None:
    """Reset persisted entries to empty+auto for the given session keys."""
    p = st.session_state.get('_persisted', {})
    for sk in session_keys:
        jk = _SESSION_TO_JSON_KEY.get(sk)
        if jk:
            p[jk] = {'name': '', 'auto_select': True}
    st.session_state['_persisted'] = p


def _on_ct_dir_change() -> None:
    _mark_path_manual('ct_dir')


def _on_rtstruct_path_change() -> None:
    _mark_path_manual('rtstruct_path')


def _on_plan_path_change() -> None:
    _mark_path_manual('plan_path')


def _browse_folder(session_key: str, label: str = 'Select folder'):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(title=label)
    root.destroy()
    if folder:
        st.session_state[session_key] = folder
        if session_key == 'data_dir':
            st.session_state['_data_dir_source'] = 'browse'
            _on_data_dir_change()
        elif session_key in _PATH_SOURCE_KEYS:
            _mark_path_manual(session_key)


def _browse_file(session_key: str, label: str = 'Select file'):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    filepath = filedialog.askopenfilename(title=label)
    root.destroy()
    if filepath:
        st.session_state[session_key] = filepath
        if session_key in _PATH_SOURCE_KEYS:
            _mark_path_manual(session_key)


def parse_projection_angles_from_filenames(file_paths):
    angles = []
    for path in file_paths:
        basename = os.path.basename(path)
        matches = re.findall(r"([-+]?[0-9]*\.?[0-9]+)", basename)
        if matches:
            angles.append(float(matches[-1]))
    if angles:
        return sorted(angles)
    return None


def _dcm_modality(filepath: str) -> str | None:
    """Return the DICOM Modality tag (lowercased) for a file, or None on failure."""
    try:
        ds = dcm.dcmread(filepath, stop_before_pixels=True)
        return ds.Modality.lower()
    except Exception:
        return None


def _autofill_from_patient(patient_dir: str):
    # Detect if patient folder contains grouped planning files under CT/Plan/Structure Set
    fractions = {}
    try:
        subdirs = sorted([d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))])
    except Exception:
        subdirs = []

    modality_roots = {}
    for sub in subdirs:
        sub_lower = sub.lower()
        if sub_lower == 'ct':
            modality_roots['ct'] = os.path.join(patient_dir, sub)
        elif sub_lower == 'plan':
            modality_roots['plan'] = os.path.join(patient_dir, sub)
        elif sub_lower in ('structure set', 'structure_set', 'structure'):
            modality_roots['rtstruct'] = os.path.join(patient_dir, sub)

    if modality_roots.get('ct') and modality_roots.get('plan') and modality_roots.get('rtstruct'):
        fraction_names = set()
        for root in modality_roots.values():
            try:
                fraction_names.update([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            except Exception:
                continue

        for fraction in sorted(fraction_names):
            entry = {'ct_dir': None, 'rtstruct_path': None, 'plan_path': None}
            candidate_ct = os.path.join(modality_roots['ct'], fraction)
            if os.path.isdir(candidate_ct):
                entry['ct_dir'] = candidate_ct

            rtstruct_dir = os.path.join(modality_roots['rtstruct'], fraction)
            if os.path.isdir(rtstruct_dir):
                for f in os.listdir(rtstruct_dir):
                    if f.lower().endswith('.dcm'):
                        fp = os.path.join(rtstruct_dir, f)
                        if _dcm_modality(fp) == 'rtstruct':
                            entry['rtstruct_path'] = fp
                            break

            plan_dir = os.path.join(modality_roots['plan'], fraction)
            if os.path.isdir(plan_dir):
                for f in os.listdir(plan_dir):
                    if f.lower().endswith('.dcm'):
                        fp = os.path.join(plan_dir, f)
                        if _dcm_modality(fp) == 'rtplan':
                            entry['plan_path'] = fp
                            break

            if any(entry.values()):
                fractions[fraction] = entry

        if fractions:
            return {'fractions': fractions}

    found = {'ct_dir': None, 'rtstruct_path': None, 'plan_path': None}
    for root, _, files in os.walk(patient_dir):
        folder_name = os.path.basename(root).lower()
        # CT: identify by folder name only.
        if not found['ct_dir'] and folder_name == 'ct':
            found['ct_dir'] = root
            continue  # skip modality checks inside CT directories
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        if not dcm_files:
            continue
        # RTSTRUCT / RTPLAN: check each file's DICOM Modality tag.
        for f in dcm_files:
            mod = _dcm_modality(os.path.join(root, f))
            if mod == 'rtstruct' and not found['rtstruct_path']:
                found['rtstruct_path'] = os.path.join(root, f)
            elif mod == 'rtplan' and not found['plan_path']:
                found['plan_path'] = os.path.join(root, f)
            if found['rtstruct_path'] and found['plan_path']:
                break
        if all(found.values()):
            break
    return found


def _load_planning_for_patient(patient: str, root_for_patients: str):
    if not patient:
        return
    patient_dir = os.path.join(root_for_patients, patient)
    autofill = _autofill_from_patient(patient_dir)
    # If multiple fractions were detected, store the candidates for user selection
    if isinstance(autofill, dict) and 'fractions' in autofill:
        st.session_state['fraction_candidates'] = autofill['fractions']
        # default selected fraction
        if 'fraction' not in st.session_state:
            keys = list(autofill['fractions'].keys())
            if keys:
                st.session_state['fraction'] = keys[0]
                _mark_path_auto('fraction')
        selected_fraction = st.session_state.get('fraction')
        fraction_entry = autofill['fractions'].get(selected_fraction, {})
        # pre-populate the path fields from the first detected fraction
        if fraction_entry.get('ct_dir') and not st.session_state.get('ct_dir'):
            st.session_state['ct_dir'] = fraction_entry['ct_dir']
            _mark_path_auto('ct_dir')
        if fraction_entry.get('rtstruct_path') and not st.session_state.get('rtstruct_path'):
            st.session_state['rtstruct_path'] = fraction_entry['rtstruct_path']
            _mark_path_auto('rtstruct_path')
        if fraction_entry.get('plan_path') and not st.session_state.get('plan_path'):
            st.session_state['plan_path'] = fraction_entry['plan_path']
            _mark_path_auto('plan_path')
        # store diagnostics listing fractions
        st.session_state['planning_diagnostics'] = {
            'patient_dir': patient_dir,
            'fractions': {k: v for k, v in autofill['fractions'].items()},
        }
    else:
        # set session values even if empty to reflect found state
        if autofill.get('ct_dir') and not st.session_state.get('ct_dir'):
            st.session_state.ct_dir = autofill['ct_dir']
            _mark_path_auto('ct_dir')
        if autofill.get('rtstruct_path') and not st.session_state.get('rtstruct_path'):
            st.session_state.rtstruct_path = autofill['rtstruct_path']
            _mark_path_auto('rtstruct_path')
        if autofill.get('plan_path') and not st.session_state.get('plan_path'):
            st.session_state.plan_path = autofill['plan_path']
            _mark_path_auto('plan_path')
        # store diagnostics in session state for main-area display
        st.session_state['planning_diagnostics'] = {
            'patient_dir': patient_dir,
            'ct_candidate': st.session_state.get('ct_dir', ''),
            'ct_exists': os.path.isdir(st.session_state.get('ct_dir', '')),
            'rtstruct_candidate': st.session_state.get('rtstruct_path', ''),
            'rtstruct_exists': os.path.isfile(st.session_state.get('rtstruct_path', '')),
            'plan_candidate': st.session_state.get('plan_path', ''),
            'plan_exists': os.path.isfile(st.session_state.get('plan_path', '')),
        }



def infer_projection_files(projections_dir):
    if not projections_dir or not os.path.isdir(projections_dir):
        return []
    files = []
    for ext in ('.tiff', '.tif', '.dcm', '.mat', '.his', '.xim', '.hnc', '.hnd'):
        files.extend(sorted(Path(projections_dir).glob(f'*{ext}')))
    return [str(p) for p in files]


def load_projection_image(filepath: str, frame_info: 'HisFrameInfo | None' = None, scale: str = 'IEC61217') -> 'np.ndarray | None':
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.his':
            pixel_data, _ = load_his(filepath, frame_info=frame_info, scale=scale)
            pixel_data = np.flip(pixel_data, axis=1)
            if pixel_data is None:
                return None
            return pixel_data.T
        elif ext == '.xim':
            pixel_data, _ = load_xim(filepath, scale=scale)
            pixel_data = np.flip(pixel_data, axis=1)
            if pixel_data is None:
                return None
            return pixel_data.T
        elif ext == '.dcm':
            ds = dcm.dcmread(filepath)
            pixel_data = ds.pixel_array.astype(np.float32)
            pixel_data = np.flip(pixel_data, axis=1)
            return pixel_data.T
        elif ext in ('.tiff', '.tif'):
            pixel_data, _ = load_tiff(filepath)
            pixel_data = np.flip(pixel_data, axis=1)
            if pixel_data is None:
                return None
            return pixel_data.T
    except Exception:
        return None
    return None


def _find_projection_files_recursive(root: str) -> list[str]:
    """Recursively search root for projection files; return sorted list."""
    if not root or not os.path.isdir(root):
        return []
    files = []
    for ext in ('.xim', '.his', '.tiff', '.tif'):
        files.extend(Path(root).rglob(f'*{ext}'))
    return sorted(str(p) for p in files)




def load_geometry_from_projections(projections_dir: str, scale: str = 'IEC61217') -> dict:
    """
    Read projection geometry parameters from the first file in projections_dir.
    Returns a dict of field values plus '_loaded': set of field names that were
    successfully extracted from the file (as opposed to being hardcoded defaults).
    """
    files = infer_projection_files(projections_dir)
    if not files:
        return {}

    ext = os.path.splitext(files[0])[1].lower()
    result: dict = {}
    loaded: set = set()

    if ext == '.xim':
        _, meta = load_xim(files[0], load_pixel_data=False, scale=scale)
        result['sid'] = meta['sid']
        result['sdd'] = meta['sdd']
        result['pixel_spacing'] = float(meta['det-spacing'][0])
        result['offset_x'] = float(meta['det-offset'][0])
        result['matrix_width'] = int(meta['det-size'][0])
        result['matrix_height'] = int(meta['det-size'][1])
        loaded.update({'sid', 'sdd', 'pixel_spacing', 'offset_x', 'matrix_width', 'matrix_height'})

    elif ext == '.his':
        _, meta = load_his(files[0], load_pixel_data=False, scale=scale)
        his_info = meta.get('his-info')
        if his_info:
            result['sid'] = meta['sid']
            result['sdd'] = meta['sdd']
            result['pixel_spacing'] = float(meta['det-spacing'][0])
            result['matrix_width'] = his_info.size_x
            result['matrix_height'] = his_info.size_y
            loaded.update({'sid', 'sdd', 'pixel_spacing', 'matrix_width', 'matrix_height'})
        det_offset = meta.get('det-offset')
        if det_offset is not None:
            result['offset_x'] = float(abs(det_offset[0]))
            loaded.add('offset_x')
            result['_frames_xml'] = str(Path(files[0]).parent / '_Frames.xml')

    elif ext in ('.hnc', '.hnd'):
        result['matrix_width'] = 1024
        result['matrix_height'] = 768
        loaded.update({'matrix_width', 'matrix_height'})

    elif ext == '.dcm':
        try:
            info = dcm.dcmread(files[0], stop_before_pixels=True)
            if hasattr(info, 'Rows') and hasattr(info, 'Columns'):
                result['matrix_width'] = int(info.Columns)
                result['matrix_height'] = int(info.Rows)
                loaded.update({'matrix_width', 'matrix_height'})
            if hasattr(info, 'ImagePlanePixelSpacing'):
                result['pixel_spacing'] = float(info.ImagePlanePixelSpacing[0])
                loaded.add('pixel_spacing')
            if hasattr(info, 'RTImageSID'):
                result['sdd'] = float(round(info.RTImageSID))
                loaded.add('sdd')
            if hasattr(info, 'RadiationMachineSAD'):
                result['sid'] = float(round(info.RadiationMachineSAD))
                loaded.add('sid')
            if hasattr(info, 'RTImagePosition'):
                result['offset_x'] = float(round(abs(info.RTImagePosition[0])))
                loaded.add('offset_x')
        except Exception:
            pass

    elif ext in ('.tiff', '.tif'):
        try:
            _, info = load_tiff(files[0])
            if 'sid' in info:
                result['sid'] = float(info['sid'])
                loaded.add('sid')
            if 'sdd' in info:
                result['sdd'] = float(info['sdd'])
                loaded.add('sdd')
            if 'det-spacing' in info:
                result['pixel_spacing'] = float(info['det-spacing'][0])
                loaded.add('pixel_spacing')
            if 'det-offset' in info:
                result['offset_x'] = float(info['det-offset'][0])
                loaded.add('offset_x')
            if 'det-size' in info:
                result['matrix_width'] = int(info['det-size'][0])
                result['matrix_height'] = int(info['det-size'][1])
                loaded.update({'matrix_width', 'matrix_height'})
        except Exception:
            pass

    result['_loaded'] = loaded
    return result


def _ct_cache_hash(ct_dir: str) -> str:
    h = hashlib.md5()
    for f in sorted(Path(ct_dir).glob('*.dcm')):
        h.update(f.name.encode())
        h.update(str(f.stat().st_mtime).encode())
    return h.hexdigest()[:16]


def load_ct(ct_dir: str, progress_callback=None, on_cache_start=None, on_cache_done=None):
    ct_hash = _ct_cache_hash(ct_dir)
    npz_path = os.path.join(ct_dir, f'ct_{ct_hash}.npz')
    if os.path.isfile(npz_path):
        loaded = np.load(npz_path)
        return loaded['data'], loaded['affine']
    ct_volume, affine = from_ct_dicom(ct_dir, progress_callback=progress_callback)
    if on_cache_start:
        on_cache_start()
    save_numpy({'data': ct_volume, 'affine': affine}, npz_path)
    if on_cache_done:
        on_cache_done()
    return ct_volume, affine


@st.cache_data(show_spinner=False)
def load_rtstruct(rtstruct_path: str):
    return list_rtstruct_regions(rtstruct_path)



@st.cache_data(show_spinner=False)
def load_plan_isocentre(plan_path: str):
    plan = dcm.dcmread(plan_path, force=False)
    beam = plan.BeamSequence[0]
    control_point = beam.ControlPointSequence[0]
    return np.asarray(control_point.IsocenterPosition, dtype=float)


def make_drr_figure(image, mask, angle, title='DRR'):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray', aspect='equal')
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = np.where(mask, 0.35, 0.0)
        ax.imshow(overlay, aspect='equal')
    ax.set_title(f'{title}  (Angle {angle:.1f}°)')
    ax.axis('off')
    return fig


def get_contour_boundary(mask: np.ndarray) -> tuple:
    m = mask.astype(bool)
    padded = np.pad(m, 1)
    eroded = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] &
        padded[1:-1, :-2] & padded[1:-1, 2:] & m
    )
    return np.where(m & ~eroded)


def _compute_onscreen_status() -> None:
    """Compute per-projection onscreen status and cache it in session state."""
    result = st.session_state.get('drr_result')
    if result is None:
        st.session_state.pop('_onscreen_status', None)
        st.session_state.pop('_onscreen_key', None)
        return
    labels = result['labels']
    pixel_spacing = float(st.session_state.get('pixel_spacing', 0.388))
    margin_width_mm = float(st.session_state.get('margin_width_mm', 10.0))
    margin_px = margin_width_mm / pixel_spacing if pixel_spacing > 0 else 0
    n = labels.shape[0]
    onscreen = []
    for i in range(n):
        mask = (labels[i][0] if labels.ndim == 4 else labels[i]).T
        img_h, img_w = mask.shape[0], mask.shape[1]
        y_bnd, x_bnd = get_contour_boundary(mask)
        if len(x_bnd) == 0 or len(y_bnd) == 0:
            onscreen.append(False)
            continue
        on = (float(np.min(x_bnd)) >= margin_px and float(np.max(x_bnd)) <= img_w - margin_px and
              float(np.min(y_bnd)) >= margin_px and float(np.max(y_bnd)) <= img_h - margin_px)
        onscreen.append(bool(on))
    st.session_state['_onscreen_status'] = onscreen
    st.session_state['_onscreen_key'] = (margin_width_mm, pixel_spacing)


def _histogram_fig(img: np.ndarray, vmin: float, vmax: float) -> go.Figure:
    flat = img.ravel()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=flat, nbinsx=128, marker_color='steelblue', showlegend=False))
    fig.add_vrect(x0=vmin, x1=vmax, fillcolor='orange', opacity=0.3, line_width=0)
    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        bargap=0,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _go_to_proj(idx: int):
    st.session_state['current_proj_idx'] = idx


def _proj_prev():
    n = len(st.session_state.drr_result['angles'])
    cur = st.session_state.get('current_proj_idx', 1)
    st.session_state['current_proj_idx'] = n if cur <= 1 else cur - 1


def _proj_next():
    n = len(st.session_state.drr_result['angles'])
    cur = st.session_state.get('current_proj_idx', 1)
    st.session_state['current_proj_idx'] = 1 if cur >= n else cur + 1


def _get_offsets_save_path() -> str:
    pdir = st.session_state.get('treatment_images', '')
    if pdir and os.path.isdir(pdir):
        return os.path.join(pdir, 'Contours', 'contour_shifts.csv')
    return str(Path.home() / 'contour_shifts.csv')


def _save_contour_offsets() -> None:
    result = st.session_state.get('drr_result')
    if result is None:
        return
    n = len(result['angles'])
    proj_files = result.get('proj_files') or []
    pixel_spacing = float(st.session_state.get('pixel_spacing', 0.388))
    save_path = Path(_get_offsets_save_path())

    # Read existing CSV so frames outside the current session are preserved.
    existing: dict = {}
    if save_path.is_file():
        try:
            for line in save_path.read_text().splitlines()[1:]:
                parts = line.split(',')
                if len(parts) >= 3:
                    conf = int(parts[3]) if len(parts) > 3 else 0
                    clin = parts[4].strip().lower() == 'true' if len(parts) > 4 else False
                    existing[parts[0].strip()] = {
                        'u': float(parts[1]), 'v': float(parts[2]),
                        'conf': conf, 'clin': clin,
                    }
        except Exception:
            pass

    # Merge current session values (overwrite matching entries).
    for i in range(n):
        fname = Path(proj_files[i]).stem if (proj_files and i < len(proj_files)) else str(i)
        ox = st.session_state.get(f'_ox_{i}', 0)
        oy = st.session_state.get(f'_oy_{i}', 0)
        conf = st.session_state.get(f'_conf_{i}', 0)
        clin = st.session_state.get(f'_clin_{i}', False)
        existing[fname] = {'u': ox * pixel_spacing, 'v': oy * pixel_spacing, 'conf': conf, 'clin': clin}

    rows = ['file,u-direction(mm),v-direction(mm),confidence,clinical']
    for fname, d in existing.items():
        rows.append(f"{fname},{d['u']:.4f},{d['v']:.4f},{d['conf']},{d['clin']}")
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('\n'.join(rows))
        st.session_state['_offsets_last_saved_path'] = str(save_path)
    except Exception:
        pass


def _load_contour_offsets(filepath: str) -> bool:
    try:
        result = st.session_state.get('drr_result')
        proj_files = (result.get('proj_files') or []) if result else []
        pixel_spacing = float(st.session_state.get('pixel_spacing', 0.388))
        stem_to_idx = {Path(fp).stem: i for i, fp in enumerate(proj_files)}
        lines = Path(filepath).read_text().splitlines()
        for line in lines[1:]:  # skip header
            parts = line.split(',')
            if len(parts) < 3:
                continue
            fname = parts[0].strip()
            u_mm, v_mm = float(parts[1]), float(parts[2])
            conf = int(parts[3]) if len(parts) > 3 else 0
            clin = parts[4].strip().lower() == 'true' if len(parts) > 4 else False
            ox = int(round(u_mm / pixel_spacing)) if pixel_spacing else 0
            oy = int(round(v_mm / pixel_spacing)) if pixel_spacing else 0
            if fname in stem_to_idx:
                i = stem_to_idx[fname]
            else:
                try:
                    i = int(fname)
                except ValueError:
                    continue
            st.session_state[f'_ox_{i}'] = ox
            st.session_state[f'_oy_{i}'] = oy
            st.session_state[f'_conf_{i}'] = conf
            st.session_state[f'_clin_{i}'] = clin
        st.session_state['_force_counter'] = st.session_state.get('_force_counter', 0) + 1
        return True
    except Exception:
        return False



@st.fragment
def render_drr_display():
    result = st.session_state.drr_result
    n_angles = len(result['angles'])

    # Navigation rendered FIRST so proj_idx is the slider's return value, not a
    # pre-render session_state read. Inside @st.fragment, widget keys are committed to
    # session_state when the widget executes — reading 'current_proj_idx' before
    # st.slider() returns the previous value, so images show the wrong projection on drag.
    _proj_col, _ = st.columns(2)
    with _proj_col:
        _pv, _ps, _pn = st.columns([1, 12, 1])
        _pv.button('◀', key='proj_prev', on_click=_proj_prev)
        with _ps:
            proj_idx = st.slider('Projection index', min_value=1, max_value=n_angles,
                                 key='current_proj_idx')
            if n_angles > 1:
                _tick_idxs = [0, (n_angles - 1) // 4, (n_angles - 1) // 2,
                              3 * (n_angles - 1) // 4, n_angles - 1]
                _all_angles_json = json.dumps([round(float(a), 1) for a in result['angles']])
                _tick_indices_json = json.dumps(_tick_idxs)
                st.markdown(f"""<script>
(function(){{
  var ALL_ANGLES={_all_angles_json};
  var TICK_IDXS={_tick_indices_json};

  // Append angle to thumb value via ::after — slider already shows 1-based index natively.
  if(!document.getElementById('_proj-angle-css')){{
    var _st=document.createElement('style');
    _st.id='_proj-angle-css';
    _st.textContent='[data-testid="stSliderThumbValue"] p::after{{content:var(--proj-angle,"");font-size:0.85em;opacity:0.75;}}';
    document.head.appendChild(_st);
  }}

  function setup(){{
    var container=null;
    var candidates=document.querySelectorAll('[data-testid="stSlider"]');
    for(var i=0;i<candidates.length;i++){{
      var lbl=candidates[i].querySelector('label');
      if(lbl&&lbl.textContent.trim()==='Projection index'){{container=candidates[i];break;}}
    }}
    if(!container)return false;
    // Streamlit uses div[role="slider"], not input[type="range"].
    var sliderDiv=container.querySelector('[role="slider"]');
    if(!sliderDiv)return false;
    if(sliderDiv._sliderSetup)return true;
    sliderDiv._sliderSetup=true;

    var trackWrap=sliderDiv.parentElement;
    var sliderBody=trackWrap.parentElement;

    // Hide Streamlit's min/max tick bar.
    var tick=sliderBody.querySelector('[data-testid="stTickBar"]');
    if(tick)tick.style.display='none';

    var PAD='0.75rem';
    var STYLE='display:flex;justify-content:space-between;padding:0 '+PAD
      +';font-size:11px;color:#888;';

    // Index labels above the track (1-based display).
    var idxBar=document.createElement('div');
    idxBar.style.cssText=STYLE+'margin-bottom:4px;';
    TICK_IDXS.forEach(function(idx){{
      var s=document.createElement('span');
      s.textContent=String(idx+1);
      idxBar.appendChild(s);
    }});
    sliderBody.insertBefore(idxBar,trackWrap);

    // Angle labels below the track at 5 static positions (1-based display).
    var angBar=document.createElement('div');
    angBar.style.cssText=STYLE+'margin-top:4px;';
    TICK_IDXS.forEach(function(idx){{
      var s=document.createElement('span');
      s.textContent=ALL_ANGLES[idx]!==undefined?(idx+1)+' ('+ALL_ANGLES[idx].toFixed(1)+'°)':String(idx+1);
      angBar.appendChild(s);
    }});
    sliderBody.insertBefore(angBar,tick||null);

    // Poll to keep the thumb showing 1-based index + angle via CSS custom property.
    if(window._projThumbInterval)clearInterval(window._projThumbInterval);
    window._projThumbInterval=setInterval(function(){{
      var _c=null,_cs=document.querySelectorAll('[data-testid="stSlider"]');
      for(var _i=0;_i<_cs.length;_i++){{
        var _l=_cs[_i].querySelector('label');
        if(_l&&_l.textContent.trim()==='Projection index'){{_c=_cs[_i];break;}}
      }}
      if(!_c)return;
      var _sd=_c.querySelector('[role="slider"]');
      var _p=_c.querySelector('[data-testid="stSliderThumbValue"] p');
      if(!_sd||!_p)return;
      var _v=parseInt(_sd.getAttribute('aria-valuenow'),10);
      var _a=ALL_ANGLES[_v-1];
      _p.style.setProperty('--proj-angle',_a!==undefined?'" ('+_a.toFixed(1)+'°)"':'""');
    }},150);
    return true;
  }}

  var n=0,t=setInterval(function(){{if(setup()||++n>40)clearInterval(t);}},100);
}})();
</script>""", unsafe_allow_html=True)
        _pn.button('▶', key='proj_next', on_click=_proj_next)

    _prospective = st.session_state.get('prospective_mode', False)

    # Save offsets whenever the projection index changes (not needed in prospective mode).
    _prev_proj_idx = st.session_state.get('_prev_proj_idx_for_save')
    st.session_state['_prev_proj_idx_for_save'] = proj_idx
    if not _prospective and _prev_proj_idx is not None and _prev_proj_idx != proj_idx:
        _save_contour_offsets()

    _i = proj_idx - 1  # 0-based index for array/state access (state is always 0-based)
    drr = np.flip(result['drrs'][_i], axis=1).T
    _labels = result['labels']
    mask = np.flip((_labels[_i][0] if _labels.ndim == 4 else _labels[_i]), axis=1).T
    angle = float(result['angles'][_i])
    offset_x = st.session_state.get(f'_ox_{_i}', 0)
    offset_y = st.session_state.get(f'_oy_{_i}', 0)
    proj_files = result.get('proj_files') or ([] if _prospective else infer_projection_files(st.session_state.get('treatment_images', '')))
    treat_img = None
    if not _prospective and proj_files and _i < len(proj_files):
        _fi = result.get('frame_infos', {}).get(os.path.basename(proj_files[_i]))
        _scale = st.session_state.get('proj_scale', 'IEC61217')
        treat_img = load_projection_image(proj_files[_i], frame_info=_fi, scale=_scale)

    # -- Hist Eq. + intensity controls (above images) --
    use_heq = st.session_state.get('use_hist_eq', True)

    drr_disp = hist_eq(drr) if use_heq else drr
    treat_disp = hist_eq(treat_img) if (use_heq and treat_img is not None) else treat_img

    def _init_intensity_key(key, lo, hi, p1, p99):
        if key not in st.session_state:
            st.session_state[key] = (p1, p99)
        else:
            v0, v1 = st.session_state[key]
            v0 = float(np.clip(v0, lo, hi))
            v1 = float(np.clip(v1, lo, hi))
            if v0 >= v1:
                v0, v1 = p1, p99
            st.session_state[key] = (v0, v1)

    d_key = 'drr_intensity_heq' if use_heq else 'drr_intensity_noheq'
    d_lo, d_hi = float(np.nanmin(drr_disp)), float(np.nanmax(drr_disp))
    d_p1, d_p99 = float(np.percentile(drr_disp, 1)), float(np.percentile(drr_disp, 99))
    d_step = max((d_hi - d_lo) / 1000, 1e-6)
    _init_intensity_key(d_key, d_lo, d_hi, d_p1, d_p99)

    t_key = 'treat_intensity_heq' if use_heq else 'treat_intensity_noheq'
    if treat_disp is not None:
        t_lo, t_hi = float(np.nanmin(treat_disp)), float(np.nanmax(treat_disp))
        t_p1, t_p99 = float(np.percentile(treat_disp, 1)), float(np.percentile(treat_disp, 99))
        t_step = max((t_hi - t_lo) / 1000, 1e-6)
        _init_intensity_key(t_key, t_lo, t_hi, t_p1, t_p99)

    if _prospective:
        _prosp_col, _ = st.columns(2)
        with _prosp_col:
            st.slider('Projection intensity', min_value=d_lo, max_value=d_hi, step=d_step, key=d_key)
            st.plotly_chart(_histogram_fig(drr_disp, *st.session_state[d_key]), width='stretch')
    else:
        col_t_int, col_d_int = st.columns(2)
        with col_t_int:
            if treat_disp is not None:
                st.slider('Treatment intensity', min_value=t_lo, max_value=t_hi,
                          step=t_step, key=t_key)
                st.plotly_chart(_histogram_fig(treat_disp, *st.session_state[t_key]), width='stretch')
        with col_d_int:
            st.slider('Projection intensity', min_value=d_lo, max_value=d_hi,
                      step=d_step, key=d_key)
            st.plotly_chart(_histogram_fig(drr_disp, *st.session_state[d_key]), width='stretch')

    # -- Images with drag-and-drop contour overlay --
    img_label = f'{proj_idx}/{n_angles}, kv det. angle={angle:.1f}°'

    y_bnd, x_bnd = get_contour_boundary(mask)
    _img_w = int(drr_disp.shape[1])
    _img_h = int(drr_disp.shape[0])
    # Slider ranges: contour can move until fully just off any edge.
    if len(x_bnd) and len(y_bnd):
        _ox_min = int(-np.max(x_bnd))
        _ox_max = int(_img_w - np.min(x_bnd))
        _oy_min = int(np.min(y_bnd) - _img_h)
        _oy_max = int(np.max(y_bnd))
    else:
        _ox_min, _ox_max, _oy_min, _oy_max = -500, 500, -500, 500

    d_vmin, d_vmax = st.session_state.get(d_key, (d_p1, d_p99))
    t_vmin, t_vmax = st.session_state.get(t_key, (0.0, 1.0)) if treat_disp is not None else (0.0, 1.0)
    treat_src = (_img_to_src(treat_disp, t_vmin, t_vmax) if treat_disp is not None
                 else _img_to_src(drr_disp, d_vmin, d_vmax))
    drr_src = _img_to_src(drr_disp, d_vmin, d_vmax)
    _zoom = st.session_state.get('_zoom', {'x0': 0.0, 'y0': 0.0, 'x1': 1.0, 'y1': 1.0})
    _show_contour = st.session_state.get('contour_visible', True)

    if _prospective:
        # Recompute onscreen status if margin or pixel spacing changed.
        _on_key = (float(st.session_state.get('margin_width_mm', 10.0)),
                   float(st.session_state.get('pixel_spacing', 0.388)))
        if st.session_state.get('_onscreen_key') != _on_key:
            _compute_onscreen_status()
        _onscreen = st.session_state.get('_onscreen_status')

        col_canvas, col_list = st.columns(2)
        with col_canvas:
            # Onscreen indicators for current projection.
            if _onscreen is not None:
                _threshold = float(st.session_state.get('onscreen_threshold', 50.0))
                _n_on = sum(_onscreen)
                _n_total = len(_onscreen)
                _pct = _n_on / _n_total * 100 if _n_total > 0 else 0
                _cur_on = _onscreen[_i] if _i < len(_onscreen) else False
                _ind_col, _tot_col = st.columns(2)
                _ind_col.markdown(
                    f'{"🟢" if _cur_on else "🔴"}&ensp;**{"Contour onscreen" if _cur_on else "Contour offscreen"}**'
                )
                _tot_col.markdown(
                    f'{"🟢" if _pct >= _threshold else "🔴"}&ensp;**Total onscreen ({_n_on}/{_n_total}, {_pct:.0f}%)**'
                )
            st.markdown(f'**Projection image** ({img_label})')
            drag_result = _contour_drag(
                treat_src=treat_src,
                drr_src=drr_src,
                cx=x_bnd.tolist() if _show_contour else [],
                cy=y_bnd.tolist() if _show_contour else [],
                offset_x=offset_x,
                offset_y=offset_y,
                force_counter=st.session_state.get('_force_counter', 0),
                zoom=_zoom,
                zoom_step=st.session_state.get('zoom_step', 10),
                contour_color=st.session_state.get('contour_color', '#ff3c3c'),
                contour_linestyle=st.session_state.get('contour_linestyle', 'solid'),
                img_w=_img_w,
                img_h=_img_h,
                prospective=_prospective,
                show_margin=True if st.session_state.get('show_margin', True) else False,
                margin_width_mm=float(st.session_state.get('margin_width_mm', 10.0)),
                pixel_spacing=float(st.session_state.get('pixel_spacing', 0.388)),
                key=f'contour_drag_{_i}',
                default=None,
                height=600,
            )
        with col_list:
            if _onscreen is not None:
                _angles_list = result['angles']
                _on_idxs  = [j for j, on in enumerate(_onscreen) if on]
                _off_idxs = [j for j, on in enumerate(_onscreen) if not on]
                # Inject CSS once to make projection-list buttons look like plain text links.
                st.markdown("""<style>
[data-proj-list] { background:none !important; border:none !important;
  box-shadow:none !important; padding:1px 0 !important; margin:0 !important;
  text-align:left !important; color:inherit !important; font-size:0.9em !important;
  line-height:1.7 !important; width:auto !important; cursor:pointer; }
[data-proj-list]:hover { text-decoration:underline !important; background:none !important; }
[data-proj-list]:focus { outline:none !important; box-shadow:none !important; }
</style>
<script>
(function(){
  if(window._projListObserver) return;
  function mark(){
    document.querySelectorAll('[data-testid="stButton"] button').forEach(function(b){
      var t=(b.innerText||'').trim();
      if(t.startsWith('•')||t.startsWith('▶')){b.setAttribute('data-proj-list','1');}
    });
  }
  window._projListObserver=new MutationObserver(mark);
  window._projListObserver.observe(document.body,{childList:true,subtree:true});
  mark();
})();
</script>""", unsafe_allow_html=True)
                st.markdown('🟢 **Onscreen**')
                for j in _on_idxs:
                    _cur = j == _i
                    _lbl = f'{"▶ " if _cur else ""}• Projection {j+1} ({_angles_list[j]:.1f}°)'
                    st.button(_lbl, key=f'_proj_on_{j}', on_click=_go_to_proj, args=(j + 1,))
                st.markdown('🔴 **Offscreen**')
                for j in _off_idxs:
                    _cur = j == _i
                    _lbl = f'{"▶ " if _cur else ""}• Projection {j+1} ({_angles_list[j]:.1f}°)'
                    st.button(_lbl, key=f'_proj_off_{j}', on_click=_go_to_proj, args=(j + 1,))
    else:
        _lc, _rc = st.columns(2)
        _lc.markdown(f'**Treatment image** ({img_label})')
        _rc.markdown(f'**Projection image** ({img_label})')
        drag_result = _contour_drag(
            treat_src=treat_src,
            drr_src=drr_src,
            cx=x_bnd.tolist() if _show_contour else [],
            cy=y_bnd.tolist() if _show_contour else [],
            offset_x=offset_x,
            offset_y=offset_y,
            force_counter=st.session_state.get('_force_counter', 0),
            zoom=_zoom,
            zoom_step=st.session_state.get('zoom_step', 10),
            contour_color=st.session_state.get('contour_color', '#ff3c3c'),
            contour_linestyle=st.session_state.get('contour_linestyle', 'solid'),
            img_w=_img_w,
            img_h=_img_h,
            prospective=False,
            show_margin=False,
            margin_width_mm=float(st.session_state.get('margin_width_mm', 10.0)),
            pixel_spacing=float(st.session_state.get('pixel_spacing', 0.388)),
            key=f'contour_drag_{_i}',
            default=None,
            height=600,
        )
    # Apply drag result in-place — sliders below pick up the updated values, no rerun needed.
    _last_drag = st.session_state.get(f'_last_drag_applied_{_i}')
    if drag_result is not None and drag_result != _last_drag:
        st.session_state[f'_last_drag_applied_{_i}'] = drag_result
        new_zoom = drag_result.get('zoom')
        if new_zoom:
            st.session_state['_zoom'] = new_zoom
        new_ox = drag_result['offset_x']
        new_oy = drag_result['offset_y']
        if new_ox != offset_x or new_oy != offset_y:
            _undo_stack = st.session_state.get(f'_undo_stack_{_i}', [])
            _undo_stack.append((offset_x, offset_y))
            st.session_state[f'_undo_stack_{_i}'] = _undo_stack
            offset_x = max(_ox_min, min(_ox_max, new_ox))
            offset_y = max(_oy_min, min(_oy_max, new_oy))
            st.session_state[f'_ox_{_i}'] = offset_x
            st.session_state[f'_oy_{_i}'] = offset_y

    # Confidence + offset panel (fixed-position overlay in parent page) — retrospective only.
    if not _prospective:
        _panel_result = _confidence_panel(
            confidence=st.session_state.get(f'_conf_{_i}', 0),
            clinical=st.session_state.get(f'_clin_{_i}', False),
            offset_x=offset_x,
            offset_y=offset_y,
            ox_min=_ox_min,
            ox_max=_ox_max,
            oy_min=_oy_min,
            oy_max=_oy_max,
            undo_available=len(st.session_state.get(f'_undo_stack_{_i}', [])) > 0,
            key='conf_panel',
            default=None,
            height=0,
        )
        if _panel_result is not None:
            _panel_event_id = _panel_result.get('event_id', 0)
            _last_panel_event_id = st.session_state.get('_last_panel_event_id', -1)
            if _panel_event_id > _last_panel_event_id:
                st.session_state['_last_panel_event_id'] = _panel_event_id
                st.session_state[f'_conf_{_i}'] = _panel_result.get('confidence', 0)
                st.session_state[f'_clin_{_i}'] = _panel_result.get('clinical', False)
                # Only update offsets when the panel explicitly sent them (offset slider/button events).
                # Confidence/clinical-only events omit these keys to avoid clobbering drag state.
                _panel_action = _panel_result.get('action')
                if _panel_action == 'undo':
                    _undo_stack = st.session_state.get(f'_undo_stack_{_i}', [])
                    if _undo_stack:
                        prev_ox, prev_oy = _undo_stack.pop()
                        st.session_state[f'_undo_stack_{_i}'] = _undo_stack
                        offset_x = prev_ox
                        offset_y = prev_oy
                        st.session_state[f'_ox_{_i}'] = offset_x
                        st.session_state[f'_oy_{_i}'] = offset_y
                        st.session_state['_force_counter'] = st.session_state.get('_force_counter', 0) + 1
                        st.rerun(scope='fragment')
                elif 'offset_x' in _panel_result or 'offset_y' in _panel_result:
                    _new_ox = _panel_result.get('offset_x', offset_x)
                    _new_oy = _panel_result.get('offset_y', offset_y)
                    if _new_ox != offset_x or _new_oy != offset_y:
                        _undo_stack = st.session_state.get(f'_undo_stack_{_i}', [])
                        _undo_stack.append((offset_x, offset_y))
                        st.session_state[f'_undo_stack_{_i}'] = _undo_stack
                        offset_x = max(_ox_min, min(_ox_max, _new_ox))
                        offset_y = max(_oy_min, min(_oy_max, _new_oy))
                        st.session_state[f'_ox_{_i}'] = offset_x
                        st.session_state[f'_oy_{_i}'] = offset_y
                        st.session_state['_force_counter'] = st.session_state.get('_force_counter', 0) + 1
                        st.rerun(scope='fragment')

    # -- Projection details --
    st.markdown('---')
    st.subheader('Projection details')
    st.write(f'* Method: {result["method"]}')
    st.write(f'* Projections: {n_angles}')
    st.write(f'* Structure: {result["region"]}')
    st.write(f'* Angle: {angle:.2f}°')
    st.write(f'* Contour offset: ({offset_x}, {offset_y}) px')

    if st.button('Save current projection as PNG', key='save_drr'):
        out_path = Path(tempfile.gettempdir()) / f'contour_alignment_drr_{_i}.png'
        fig_save = make_drr_figure(drr_disp, mask, angle, title=f'Contour overlay: {result["region"]}')
        fig_save.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig_save)
        st.success(f'Saved projection to {out_path}')



def _log_drr_section(patients, root_for_patients):
    st.markdown("""<style>
label[role="switch"][aria-checked="true"] > div:first-of-type { background-color: #28a745 !important; }
</style>""", unsafe_allow_html=True)
    if not st.toggle('Logging', value=True, key='show_logging'):
        return
    st.markdown('---')

    _state_path = st.session_state.get('_state_file_loaded')
    if _state_path:
        st.markdown(f'🟢&ensp;Loaded session state from: `{_state_path}`')

    _log_data_dir_section(patients, root_for_patients)
    _log_patient_section()
    _log_fraction_section(root_for_patients)
    _log_post_finalise_section()

    _saved_planning = st.session_state.get('_state_saved_planning')
    if _saved_planning:
        st.markdown(f'🟢&ensp;Saved session state to: `{_saved_planning}`')
        st.markdown(f'🟢&ensp;Session state: `{str(st.session_state.get("_persisted", {})).replace(chr(92)*2, chr(92))}`')

    result = st.session_state.get('drr_result')

    _prospective = st.session_state.get('prospective_mode', False)

    # Projection files — only show after planning data has been finalised (retrospective only).
    if st.session_state.get('planning_finalised') and not _prospective:
        _proj_files = infer_projection_files(st.session_state.get('treatment_images', ''))
        if _proj_files:
            _s = 's' if len(_proj_files) != 1 else ''
            st.markdown(f'🟢&ensp;**{len(_proj_files)}** treatment projection file{_s} found. First five:\n' + '\n'.join(f'- `{f}`' for f in _proj_files[:5]))
        else:
            st.markdown('🔴&ensp;No treatment projection files found')

    # Projection angles — shown before and after create.
    if st.session_state.get('planning_finalised'):
        if result is not None:
            _log_angles = result['angles']
            st.markdown(f'🟢&ensp;Projection angles ({len(_log_angles)}): `{[round(a, 2) for a in _log_angles]}`')
        else:
            _n_ant = int(st.session_state.get('n_proj', 8))
            _start_ant = float(st.session_state.get('angle_start', 0.0))
            _stop_ant = float(st.session_state.get('angle_stop', 359.0))
            _anticipated = [round(a, 2) for a in np.linspace(_start_ant, _stop_ant, _n_ant)]
            st.markdown(f'🟢&ensp;Projection angles ({_n_ant}): `{_anticipated}`')

    if result is None:
        return

    n = len(result['angles'])

    if not _prospective:
        _csv_path = _get_offsets_save_path()
        if Path(_csv_path).is_file():
            st.markdown(f'🟢&ensp;Loading contour offsets from: `{_csv_path}`')

    # ── Post-projection section (rerenders on projection index change) ────────
    st.markdown('---')

    if not _prospective:
        # Per-projection contour offsets.
        _px_sp = float(st.session_state.get('pixel_spacing', 0.388))
        _offsets_px = {i+1: (st.session_state.get(f'_ox_{i}', 0),
                             st.session_state.get(f'_oy_{i}', 0))
                       for i in range(n)}
        _offsets_mm = {i+1: (round(st.session_state.get(f'_ox_{i}', 0) * _px_sp, 4),
                             round(st.session_state.get(f'_oy_{i}', 0) * _px_sp, 4))
                       for i in range(n)}
        st.markdown(f'🟢&ensp;Contour offsets ({len(_offsets_px)}, px): `{_offsets_px}`')
        st.markdown(f'🟢&ensp;Contour offsets ({len(_offsets_mm)}, mm): `{_offsets_mm}`')

        _conf_scores = {i+1: st.session_state.get(f'_conf_{i}', 0) for i in range(n)}
        _clin_flags  = {i+1: st.session_state.get(f'_clin_{i}', False) for i in range(n)}
        st.markdown(f'🟢&ensp;Confidence scores (0–5): `{_conf_scores}`')
        st.markdown(f'🟢&ensp;Clinically acceptable: `{_clin_flags}`')

        _last_saved = st.session_state.get('_offsets_last_saved_path')
        _csv_path = _get_offsets_save_path()
        st.markdown(f'🟢&ensp;Saved contour offsets to: `{_last_saved or _csv_path}`')

    _saved_proj = st.session_state.get('_state_saved_projections')
    if _saved_proj:
        st.markdown(f'🟢&ensp;Saved session state to: `{_saved_proj}`')
        st.markdown(f'🟢&ensp;Session state: `{str(st.session_state.get("_persisted", {})).replace(chr(92)*2, chr(92))}`')


def validate_directory(path: str):
    return path and os.path.isdir(path)


def validate_file(path: str):
    return path and os.path.isfile(path)


_PLANS_OPTIONS = ["PatientPlans", "Patient Plans", "PlanningFiles", "Planning Files"]
_IMAGES_OPTIONS = ["Patient Images", "PatientImages", "Treatment Images", "TreatmentImages"]


def _on_proj_machine_change() -> None:
    if st.session_state.get('prospective_mode', False):
        _pfs = st.session_state.get('_prosp_fields_set') or set()
        _pfs.add('proj_machine')
        st.session_state['_prosp_fields_set'] = _pfs


def _on_prospective_change() -> None:
    val = st.session_state.get('prospective_mode', False)
    p = st.session_state.get('_persisted', {})
    p['prospective'] = {'name': bool(val), 'auto_select': True, 'depends_on': 'patient'}
    st.session_state['_persisted'] = p
    _save_persisted_state()


def _on_data_dir_change():
    for key in ('patient', 'ct_dir', 'rtstruct_path', 'plan_path', 'fraction_candidates',
                'fraction', 'planning_diagnostics', 'loaded_regions', 'loaded_isocentre',
                'drr_result', 'planning_finalised',
                'geometry_from_projections', '_prev_treatment_images',
                'treat_fraction', 'treat_imaging_type', 'cbct_session', 'treatment_images',
                'prospective_mode'):
        st.session_state.pop(key, None)
    _clear_persisted('patient', 'ct_dir', 'plan_path', 'rtstruct_path', 'selected_region',
                     'treat_fraction', 'treat_imaging_type', 'cbct_session', 'treatment_images',
                     'prospective_mode')
    _save_persisted_state()


def _on_patient_change():
    _mark_path_manual('patient')
    for key in ('ct_dir', 'rtstruct_path', 'plan_path', 'fraction_candidates', 'fraction',
                'planning_diagnostics', 'loaded_regions', 'loaded_isocentre', 'drr_result',
                'planning_finalised',
                'geometry_from_projections', '_prev_treatment_images',
                'prospective_mode'):
        st.session_state.pop(key, None)
    _clear_persisted('ct_dir', 'plan_path', 'rtstruct_path', 'selected_region', 'prospective_mode')
    _save_persisted_state()
    _root = st.session_state.get('data_dir', '')
    for opt in _PLANS_OPTIONS:
        candidate = os.path.join(_root, opt)
        if os.path.isdir(candidate):
            _root = candidate
            break
    _load_planning_for_patient(st.session_state.get('patient'), _root)


def _inject_field_styles(fields: list) -> None:
    rules = []
    for label, color, *_ in fields:
        if not color:
            continue
        le = label.replace("'", "\\'")
        # Text inputs: border lives on the inner [data-baseweb='input'] div
        rules.append(
            f"[data-testid='stTextInput']:has(input[aria-label='{le}']) [data-baseweb='input'] {{"
            f"border-color:{color} !important;"
            f"box-shadow:0 0 0 1px {color} !important;}}"
        )
        # Number inputs: border lives on stNumberInputContainer itself
        rules.append(
            f"[data-testid='stNumberInputContainer']:has(input[aria-label='{le}']) {{"
            f"border-color:{color} !important;"
            f"box-shadow:0 0 0 1px {color} !important;}}"
        )
        # Selectboxes: border lives on [data-baseweb='select']
        rules.append(
            f"[data-testid='stSelectbox']:has(input[aria-label='{le}']) [data-baseweb='select'] {{"
            f"border-color:{color} !important;"
            f"box-shadow:0 0 0 1px {color} !important;}}"
        )
    if rules:
        st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def _path_color(value: str, validator) -> str:
    if not value:
        return 'red'
    return 'green' if validator(value) else 'red'


def _on_imaging_type_change() -> None:
    _mark_path_manual('treat_imaging_type')
    st.session_state.treatment_images = ''
    st.session_state.pop('cbct_session', None)
    _clear_persisted('treatment_images', 'cbct_session')
    st.session_state.pop('geometry_from_projections', None)
    st.session_state.pop('_prev_treatment_images', None)


def _on_treat_fraction_change() -> None:
    _mark_path_manual('treat_fraction')
    st.session_state['fraction'] = st.session_state.get('treat_fraction', '')
    st.session_state.treatment_images = ''
    _clear_persisted('treatment_images')
    st.session_state.pop('geometry_from_projections', None)
    st.session_state.pop('_prev_treatment_images', None)


def _on_plan_fraction_change() -> None:
    _mark_path_manual('fraction')
    for key in ('planning_finalised', 'loaded_regions', 'loaded_isocentre', 'drr_result'):
        st.session_state.pop(key, None)
    _sel_frac = st.session_state.get('fraction')
    _entry = st.session_state.get('fraction_candidates', {}).get(_sel_frac, {})
    if _entry.get('ct_dir'):
        st.session_state['ct_dir'] = _entry['ct_dir']
        _mark_path_auto('ct_dir')
    if _entry.get('rtstruct_path'):
        st.session_state['rtstruct_path'] = _entry['rtstruct_path']
        _mark_path_auto('rtstruct_path')
    if _entry.get('plan_path'):
        st.session_state['plan_path'] = _entry['plan_path']
        _mark_path_auto('plan_path')


def _on_cbct_session_change() -> None:
    _mark_path_manual('cbct_session')
    st.session_state.treatment_images = ''
    _clear_persisted('treatment_images')


def _on_treatment_images_change() -> None:
    _mark_path_manual('treatment_images')
    st.session_state.pop('geometry_from_projections', None)
    st.session_state.pop('_prev_treatment_images', None)


def _pick_default_region(regions: list) -> str | None:
    if not regions:
        return None
    upper = [r.upper() for r in regions]
    for r, u in zip(regions, upper):
        if u == 'GTV':
            return r
    for r, u in zip(regions, upper):
        if 'GTV' in u:
            return r
    for r, u in zip(regions, upper):
        if u == 'PTV':
            return r
    for r, u in zip(regions, upper):
        if 'PTV' in u:
            return r
    return sorted(regions)[0]


@st.fragment
def _log_data_dir_section(patients, root_for_patients):
    _data_dir = st.session_state.get('data_dir', '')
    if not _data_dir:
        return
    _src = st.session_state.get('_data_dir_source', '')
    if _src == 'session':
        st.markdown(f'🟢&ensp;Site folder `{_data_dir}` loaded from old session')
    else:
        st.markdown(f'🟢&ensp;Site folder set: `{_data_dir}`')
    if patients:
        _s = 's' if len(patients) != 1 else ''
        st.markdown(f'🟢&ensp;Found **{len(patients)}** patient{_s} in `{root_for_patients}`')
    else:
        st.markdown(f'🔴&ensp;No patients found in `{_data_dir}`')


@st.fragment
def _log_patient_section():
    _patient = st.session_state.get('patient', '')
    if not _patient:
        return
    st.markdown(f'🟢&ensp;Selected patient **{_patient}**')


@st.fragment
def _log_fraction_section(root_for_patients):
    _patient = st.session_state.get('patient', '')
    if not _patient:
        return
    _fracs    = st.session_state.get('fraction_candidates', {})
    _fraction = st.session_state.get('fraction', '')
    _ct = st.session_state.get('ct_dir', '')
    _rs = st.session_state.get('rtstruct_path', '')
    _rp = st.session_state.get('plan_path', '')
    if _fracs:
        _fnames = list(_fracs.keys())
        _s = 's' if len(_fnames) != 1 else ''
        _frac_root = os.path.join(root_for_patients, _patient)
        st.markdown(f'🟢&ensp;Found **{len(_fnames)}** fraction{_s} in `{_frac_root}`')
        if _fraction:
            st.markdown(f'🟢&ensp;Selected fraction **{_fraction}**')
    _patient_dir = os.path.join(root_for_patients, _patient)
    _persisted = st.session_state.get('_persisted', {})
    for _lbl, _val, _vfn, _skey in [
        ('CT folder',     _ct, validate_directory, 'ct_dir'),
        ('RTSTRUCT file', _rs, validate_file,      'rtstruct_path'),
        ('RTPLAN file',   _rp, validate_file,      'plan_path'),
    ]:
        if _val:
            _ok = _vfn(_val)
            _icon = '🟢' if _ok else '🔴'
            _jk = _SESSION_TO_JSON_KEY.get(_skey, _skey)
            _is_auto = _persisted.get(_jk, {}).get('auto_select', True)
            if not _ok:
                _verb = 'Not found at'
            elif not _is_auto:
                _verb = 'Selected'
            else:
                _verb = 'Found'
            st.markdown(f'{_icon}&ensp;{_verb} **{_lbl}**: `{_val}`')
        else:
            st.markdown(f'🔴&ensp;**{_lbl}** not found after searching `{_patient_dir}`')


@st.fragment
def _log_post_finalise_section():
    if not st.session_state.get('planning_finalised'):
        return

    _regions = st.session_state.get('loaded_regions', [])
    if _regions:
        _s = 's' if len(_regions) != 1 else ''
        st.markdown(f'🟢&ensp;Found **{len(_regions)}** RTSTRUCT region{_s}')

    _iso = st.session_state.get('loaded_isocentre')
    if _iso is not None:
        st.markdown(f'🟢&ensp;Found treatment isocentre: **({_iso[0]:.1f}, {_iso[1]:.1f}, {_iso[2]:.1f})** mm')
    else:
        st.markdown(f'🟠&ensp;No isocentre in RTPLAN — will use CT volume centre')

    _data_dir = st.session_state.get('data_dir', '')
    _patient  = st.session_state.get('patient', '')
    _images_root = None
    if validate_directory(_data_dir):
        for _opt in _IMAGES_OPTIONS:
            _cand = os.path.join(_data_dir, _opt)
            if os.path.isdir(_cand):
                _images_root = _cand
                break
    _pat_images_dir = None
    if _images_root and _patient:
        _cand = os.path.join(_images_root, _patient)
        if os.path.isdir(_cand):
            _pat_images_dir = _cand

    if _pat_images_dir:
        _img_fracs = sorted(
            d for d in os.listdir(_pat_images_dir)
            if os.path.isdir(os.path.join(_pat_images_dir, d))
        )
        if _img_fracs:
            _s = 's' if len(_img_fracs) != 1 else ''
            st.markdown(f'🟢&ensp;Found **{len(_img_fracs)}** treatment fraction{_s} in `{_pat_images_dir}`')
            _treat_frac = st.session_state.get('fraction', _img_fracs[0])
            if _treat_frac in _img_fracs:
                st.markdown(f'🟢&ensp;Selected treatment fraction **{_treat_frac}**')
                _fx_dir = os.path.join(_pat_images_dir, _treat_frac)
                _img_types = [t for t in ['CBCT', 'KIM-KV'] if os.path.isdir(os.path.join(_fx_dir, t))]
                if _img_types:
                    _s = 's' if len(_img_types) != 1 else ''
                    st.markdown(f'🟢&ensp;Found **{len(_img_types)}** imaging type{_s}: {", ".join(f"**{t}**" for t in _img_types)}')
                    _treat_type = st.session_state.get('treat_imaging_type', _img_types[0])
                    if _treat_type in _img_types:
                        st.markdown(f'🟢&ensp;Selected imaging type **{_treat_type}**')
                        _type_dir = os.path.join(_fx_dir, _treat_type)
                        if _treat_type == 'CBCT':
                            _sessions = sorted(
                                d for d in os.listdir(_type_dir)
                                if os.path.isdir(os.path.join(_type_dir, d))
                            )
                            if _sessions:
                                _s = 's' if len(_sessions) != 1 else ''
                                st.markdown(f'🟢&ensp;Found **{len(_sessions)}** CBCT session{_s} in `{_type_dir}`')
                                _session = st.session_state.get('cbct_session', _sessions[0])
                                if _session in _sessions:
                                    st.markdown(f'🟢&ensp;Selected CBCT session **{_session}**')
                else:
                    st.markdown(f'🔴&ensp;No CBCT or KIM-KV folders found for fraction **{_treat_frac}**')

    _pdir = st.session_state.get('treatment_images', '')
    _pdir_files = infer_projection_files(_pdir) if _pdir else []
    if _pdir:
        _n = len(_pdir_files)
        if _n:
            _ext = os.path.splitext(_pdir_files[0])[1].lower()
            _s = 's' if _n != 1 else ''
            st.markdown(f'🟢&ensp;Found **{_n}** `{_ext}` treatment image{_s} in `{_pdir}`')
        else:
            st.markdown(f'🔴&ensp;No treatment image files found in `{_pdir}`')
    elif st.session_state.get('fraction'):
        _treat_frac = st.session_state.get('fraction', '')
        _treat_type = st.session_state.get('treat_imaging_type', 'CBCT')
        _cbct_session = st.session_state.get('cbct_session', '')
        if _pat_images_dir and _treat_frac:
            _fx_dir = os.path.join(_pat_images_dir, _treat_frac)
            _type_dir = os.path.join(_fx_dir, _treat_type)
            _search_root = os.path.join(_type_dir, _cbct_session) if _treat_type == 'CBCT' and _cbct_session else _type_dir
        else:
            _search_root = _pat_images_dir or ''
        st.markdown(f'🔴&ensp;No `.xim`/`.his` files found after recursive search in `{_search_root}`')

    _geo = st.session_state.get('geometry_from_projections', {})
    _loaded = _geo.get('_loaded', set())
    if _loaded:
        _src_file = _pdir_files[0] if _pdir_files else None
        _geo_parts = []
        for _glabel, _gkey, _gunit in [
            ('SID', 'sid', 'mm'), ('SDD', 'sdd', 'mm'),
            ('pixel spacing', 'pixel_spacing', 'mm'),
            ('offset X', 'offset_x', 'mm'),
            ('width', 'matrix_width', 'px'), ('height', 'matrix_height', 'px'),
        ]:
            if _gkey in _loaded and _gkey in _geo:
                _gv = _geo[_gkey]
                if isinstance(_gv, float) and _gv == int(_gv):
                    _gv = int(_gv)
                _geo_parts.append(f'{_glabel}: **{_gv} {_gunit}**')
        if _geo_parts:
            _src_label = f'`{_src_file}`' if _src_file else 'treatment image'
            st.markdown(f'🟢&ensp;Loaded geometry from {_src_label}: ' + ',&ensp; '.join(_geo_parts))
    _frames_xml = _geo.get('_frames_xml')
    if _frames_xml:
        st.markdown(f'🟢&ensp;Offset X loaded from `{_frames_xml}`')
    if _pdir_files:
        _ext = os.path.splitext(_pdir_files[0])[1].lower()
        _expected_frames_xml = os.path.join(_pdir, '_Frames.xml') if _ext == '.his' else None
        _x_missing = 'offset_x' not in _loaded
        _y_missing = 'offset_y' not in _loaded
        if _expected_frames_xml:
            if os.path.isfile(_expected_frames_xml):
                _offset_src = f", missing 'Frames' element in: `{_expected_frames_xml}`"
            else:
                _offset_src = f', missing file: `{_expected_frames_xml}`'
        else:
            _offset_src = ' from projection files'
        if _x_missing and _y_missing:
            st.markdown(f'🔴&ensp;**Detector offsets** could not be loaded{_offset_src}')
        elif _x_missing:
            st.markdown(f'🔴&ensp;**Detector offset X** could not be loaded{_offset_src}')
        elif _y_missing:
            st.markdown(f'🔴&ensp;**Detector offset Y** could not be loaded{_offset_src}')
        for _glabel, _gkey in [
            ('Detector pixel spacing', 'pixel_spacing'),
            ('SID', 'sid'),
            ('SDD', 'sdd'),
            ('Detector width', 'matrix_width'),
            ('Detector height', 'matrix_height'),
        ]:
            if _gkey not in _loaded:
                st.markdown(f'🔴&ensp;**{_glabel}** could not be loaded from projection files')


def render_header():
    st.set_page_config(layout='wide', page_title='Contour Alignment Tool')
    _load_persisted_state()
    # Prevent Enter key from submitting any Streamlit form.
    st.markdown("""<script>
(function(){
  if(window._noFormEnter)return;
  window._noFormEnter=true;
  document.addEventListener('keydown',function(e){
    if(e.key!=='Enter')return;
    var tag=e.target?e.target.tagName:'';
    if(tag==='TEXTAREA'||tag==='BUTTON')return;
    if(e.target&&e.target.closest('[data-testid="stForm"]'))e.preventDefault();
  },true);
})();
</script>""", unsafe_allow_html=True)


render_header()



def _on_img_settings_change():
    _save_persisted_state()
    st.session_state['_img_settings_changed'] = True


@st.fragment
def _render_image_settings():
    if 'drr_result' not in st.session_state:
        return
    if st.session_state.pop('_img_settings_changed', False):
        st.rerun(scope='app')
    st.markdown('---')
    st.header('Image settings')
    st.toggle('Histogram equalisation', value=True, key='use_hist_eq',
              on_change=_on_img_settings_change,
              help='Histogram equalisation flattens peaks in the intensity distribution to enhance contrast.')
    st.number_input('Zoom step (%)', min_value=5, max_value=50, value=10, step=5, key='zoom_step',
                    on_change=_on_img_settings_change)
    st.toggle('Show contour', value=True, key='contour_visible', on_change=_on_img_settings_change)
    st.selectbox('Contour linestyle', ['Solid', 'Dashed', 'Dotted'], key='contour_linestyle',
                 on_change=_on_img_settings_change)
    st.color_picker('Contour colour', value='#ff3c3c', key='contour_color', on_change=_on_img_settings_change)
    if st.session_state.get('prospective_mode', False):
        st.toggle('Show margin', value=True, key='show_margin', on_change=_on_img_settings_change)
        st.number_input('Margin width (mm)', value=st.session_state.get('margin_width_mm', 10.0),
                        min_value=0.0, step=1.0, format='%.1f', key='margin_width_mm',
                        on_change=_on_img_settings_change)
        st.number_input('Onscreen threshold (%)', value=st.session_state.get('onscreen_threshold', 50.0),
                        min_value=0.0, max_value=100.0, step=5.0, format='%.0f', key='onscreen_threshold',
                        on_change=_on_img_settings_change)


@st.fragment
def _render_proj_geometry():
    if not st.session_state.get('planning_finalised'):
        return
    st.markdown('---')
    st.header('Projection geometry')

    _prospective = st.session_state.get('prospective_mode', False)

    # Detect prospective mode toggle so we can reset tracking state when it changes.
    _prosp_prev = st.session_state.get('_prosp_geom_prev')
    if _prosp_prev != _prospective:
        st.session_state['_prosp_geom_prev'] = _prospective
        st.session_state.pop('_prosp_prev_machine', None)
        st.session_state.pop('_prosp_fields_set', None)
        st.session_state.pop('_n_proj_pdir', None)

    # Machine selectbox rerenders this fragment (needed to rebuild coord-system options).
    if not _prospective:
        _proj_files_detect = infer_projection_files(st.session_state.get('treatment_images', ''))
        _has_his = any(f.lower().endswith('.his') for f in _proj_files_detect)
        _has_xim = any(f.lower().endswith('.xim') for f in _proj_files_detect)
        _auto_machine = 'Elekta' if (_has_his and not _has_xim) else ('Varian' if (_has_xim and not _has_his) else None)
        _curr_pdir_m = st.session_state.get('treatment_images', '')
        if _auto_machine and st.session_state.get('_proj_machine_auto_dir') != _curr_pdir_m:
            st.session_state['_proj_machine_auto_dir'] = _curr_pdir_m
            st.session_state['proj_machine'] = _auto_machine

    st.selectbox('Machine', ['Elekta', 'Varian'], key='proj_machine',
                 on_change=_on_proj_machine_change)
    _machine = st.session_state.get('proj_machine', 'Elekta')
    _coord_systems = ['IEC61217', 'VARIAN_IEC', 'VARIAN_STANDARD'] if _machine == 'Varian' else ['IEC61217', 'ELEKTA_IEC']
    if st.session_state.get('proj_scale') not in _coord_systems:
        st.session_state['proj_scale'] = 'IEC61217'

    # In prospective mode, apply machine defaults whenever the machine changes.
    if _prospective:
        _prev_machine = st.session_state.get('_prosp_prev_machine')
        if _prev_machine != _machine:
            st.session_state['_prosp_prev_machine'] = _machine
            if _machine == 'Elekta':
                st.session_state['sdd'] = 1536.0
                st.session_state['sid'] = 1000.0
                st.session_state['offset_x'] = 160.0
                st.session_state['offset_y'] = 0.0
                st.session_state['detector_size'] = '512x512'
                st.session_state['pixel_spacing'] = 0.8
            else:  # Varian
                st.session_state['sdd'] = 1500.0
                st.session_state['sid'] = 1000.0
                st.session_state['offset_x'] = 0.0
                st.session_state['offset_y'] = 0.0
                st.session_state['detector_size'] = '1024x768'
                st.session_state['pixel_spacing'] = 0.388
            st.session_state['_prosp_fields_set'] = {'sdd', 'sid', 'offset_x', 'pixel_spacing', 'detector_size'}

    # Pre-compute before the form so session state is initialised before widgets render.
    _proj_count = len(infer_projection_files(st.session_state.get('treatment_images', ''))) if not _prospective else 0
    _pdir = st.session_state.get('treatment_images', '') if not _prospective else ''
    _default_n_proj = 35 if _prospective else (_proj_count if _proj_count > 0 else 8)
    # Reset n_proj whenever the projections folder or prospective mode changes.
    if st.session_state.get('_n_proj_pdir') != (_pdir, _prospective):
        st.session_state['_n_proj_pdir'] = (_pdir, _prospective)
        st.session_state['n_proj'] = _default_n_proj
        st.session_state['_angles_from_files'] = None  # Unknown until next Create attempt.
    st.session_state.setdefault('n_proj', _default_n_proj)
    st.session_state.setdefault('angle_start', 0.0)
    st.session_state.setdefault('angle_stop', 359.0)
    st.session_state.setdefault('detector_size', '1024x768')
    st.session_state.setdefault('pixel_spacing', 0.388)
    st.session_state.setdefault('sid', 1000.0)
    st.session_state.setdefault('sdd', 1500.0)
    st.session_state.setdefault('offset_x', 0.0)
    st.session_state.setdefault('offset_y', 0.0)

    # Everything below is in a form: edits are buffered and nothing rerenders until
    # the user clicks "Create projections".
    with st.form('proj_geometry_form', border=False):
        st.selectbox('Coordinate system', _coord_systems, key='proj_scale',
                     help='Angles in projection files may follow different coordinate systems. '
                          'If the projections do not match the treatment images, please try a different coordinate system.')
        st.selectbox('Imaging type', ['kV', 'Megavoltage', 'CBCT'], key='imaging_type')
        _n_proj_label = f'Number of projections' if _prospective else f'Number of projections (total={_proj_count})'
        st.number_input(_n_proj_label, min_value=1, step=1, key='n_proj')
        if _prospective:
            st.number_input('Detector start angle (°)', format='%.1f', step=1.0, key='angle_start')
            st.number_input('Detector stop angle (°)', format='%.1f', step=1.0, key='angle_stop')
        _det_options = list(_DETECTOR_SIZES.keys())
        _det_cur = st.session_state.get('detector_size', '1024x768')
        if _det_cur not in _det_options:
            _det_cur = '1024x768'
        st.selectbox('Detector size', _det_options,
                     index=_det_options.index(_det_cur), key='detector_size')
        st.number_input('Detector pixel spacing (mm)', format='%.3f', key='pixel_spacing')
        st.number_input('Source-to-isocentre distance (SID, mm)', step=10.0, key='sid')
        st.number_input('Source-to-detector distance (SDD, mm)', step=10.0, key='sdd')
        st.number_input('Detector offset X (mm)', format='%.3f', key='offset_x')
        st.number_input('Detector offset Y (mm)', format='%.3f', key='offset_y')
        st.markdown('---')
        st.header('Projections')
        _sidebar_regions = st.session_state.get('loaded_regions', [])
        if _sidebar_regions:
            _cur_region = st.session_state.get('selected_region')
            if _cur_region not in _sidebar_regions:
                _cur_region = _pick_default_region(_sidebar_regions)
            _region_idx = _sidebar_regions.index(_cur_region) if _cur_region in _sidebar_regions else 0
            st.selectbox('Region to project', _sidebar_regions, index=_region_idx, key='selected_region')
        else:
            st.caption('Set RTSTRUCT path to populate regions.')
        st.selectbox('Projection method', ['DiffDRR', 'CTorch', 'IGT'], index=1, key='method')
        if st.form_submit_button('Create projections'):
            _prev_region = st.session_state.get('_prev_selected_region')
            _cur_region = st.session_state.get('selected_region')
            if _prev_region is not None and _cur_region != _prev_region:
                _mark_path_manual('selected_region')
            st.session_state['_prev_selected_region'] = _cur_region
            # Mark all geometry fields set (user confirmed via form submission).
            if _prospective:
                st.session_state['_prosp_fields_set'] = {
                    'sdd', 'sid', 'offset_x', 'offset_y', 'pixel_spacing', 'detector_size',
                }
            st.session_state['_create_triggered'] = True
            st.rerun(scope='app')

    # Highlight start/stop angle fields: orange in prospective (manual entry required),
    # red in retrospective when angles couldn't be loaded from files.
    _angle_color = 'orange' if _prospective else ('red' if st.session_state.get('_angles_from_files') is False else '')
    if _angle_color:
        _inject_field_styles([
            ('Detector start angle (°)', _angle_color),
            ('Detector stop angle (°)', _angle_color),
        ])
    # Machine dropdown: orange in prospective until user explicitly selects.
    if _prospective:
        _pfs = st.session_state.get('_prosp_fields_set') or set()
        _inject_field_styles([('Machine', 'green' if 'proj_machine' in _pfs else 'orange')])


@st.fragment
def render_sidebar():
    st.header('Patients')

    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = ''
    st.text_input('Site folder', key='data_dir', on_change=_on_data_dir_change,
                  help='Select the clinical trial folder - must follow the LEARN folder structure. Otherwise, leave empty and select planning data and treatment images individually.')
    st.button('Browse data folder', on_click=_browse_folder, args=('data_dir', 'Select patient data folder'))

    _root = st.session_state.get('data_dir', '')
    _patients = []
    _root_for_patients = _root
    if validate_directory(_root):
        for opt in _PLANS_OPTIONS:
            _cand = os.path.join(_root, opt)
            if os.path.isdir(_cand):
                _root_for_patients = _cand
                break
        _patients = sorted([d for d in os.listdir(_root_for_patients)
                             if os.path.isdir(os.path.join(_root_for_patients, d))])
    if _patients:
        _first_load = 'patient' not in st.session_state
        if _first_load:
            st.session_state.patient = _patients[0]
            _mark_path_auto('patient')
        elif st.session_state.get('patient') not in _patients:
            st.session_state.patient = _patients[0]
            _mark_path_auto('patient')
        st.selectbox('Patient', _patients, key='patient',
                     index=_patients.index(st.session_state.get('patient', _patients[0])),
                     on_change=_on_patient_change)
        if _first_load or 'fraction_candidates' not in st.session_state:
            _load_planning_for_patient(st.session_state.patient, _root_for_patients)
    else:
        st.info('Select a patient data folder to populate patients.')

    st.markdown('---')
    st.header('Planning data')

    if not st.session_state.get('data_dir', ''):
        st.caption('Set a site folder to auto-detect planning data.')

    if 'fraction_candidates' in st.session_state:
        _fracs = list(st.session_state['fraction_candidates'].keys())
        if 'fraction' not in st.session_state:
            st.session_state['fraction'] = _fracs[0]
            _mark_path_auto('fraction')
        st.selectbox('Fraction', _fracs,
                     index=_fracs.index(st.session_state.get('fraction', _fracs[0])),
                     key='fraction',
                     on_change=_on_plan_fraction_change)

    st.session_state.setdefault('ct_dir', '')
    st.session_state.setdefault('rtstruct_path', '')
    st.session_state.setdefault('plan_path', '')

    st.text_input('CT folder', key='ct_dir', on_change=_on_ct_dir_change)
    st.button('Browse CT folder', on_click=_browse_folder, args=('ct_dir', 'Select CT folder'))
    st.text_input('RTSTRUCT file', key='rtstruct_path', on_change=_on_rtstruct_path_change)
    st.button('Browse RTSTRUCT file', on_click=_browse_file, args=('rtstruct_path', 'Select RTSTRUCT file'))
    st.text_input('RTPLAN file', key='plan_path', on_change=_on_plan_path_change)
    st.button('Browse RTPLAN file', on_click=_browse_file, args=('plan_path', 'Select RTPLAN file'))

    if st.button('Finalise planning data'):
        _mark_path_manual('ct_dir')
        _mark_path_manual('rtstruct_path')
        _mark_path_manual('plan_path')
        _rp = st.session_state.get('rtstruct_path', '')
        _pp = st.session_state.get('plan_path', '')
        if validate_file(_rp):
            try:
                with st.spinner('Loading RTSTRUCT...'):
                    st.session_state.loaded_regions = load_rtstruct(_rp)
            except Exception as e:
                st.warning(f'Could not load RTSTRUCT regions: {e}')
        if validate_file(_pp):
            try:
                st.session_state.loaded_isocentre = load_plan_isocentre(_pp)
            except Exception:
                st.session_state.loaded_isocentre = None
        st.session_state['planning_finalised'] = True
        _plan_fx = st.session_state.get('fraction', '')
        if _plan_fx:
            st.session_state['treat_fraction'] = _plan_fx
            _p = st.session_state.get('_persisted', {})
            _p.setdefault('fraction', {})['name'] = _plan_fx
            st.session_state['_persisted'] = _p
        _save_persisted_state(log_key='_state_saved_planning')

    st.markdown('---')
    if not st.session_state.get('planning_finalised'):
        st.caption('Set paths above and click "Finalise planning data" to continue.')
    else:
        st.markdown("""<style>
label[role="switch"][aria-checked="true"] > div:first-of-type { background-color: #28a745 !important; }
</style>""", unsafe_allow_html=True)
        st.toggle('Prospective', value=False, key='prospective_mode',
                  on_change=_on_prospective_change,
                  help='In prospective mode there are no treatment images — projection geometry is entered manually.')
        if not st.session_state.get('prospective_mode', False):
            st.header('Treatment images')

            _data_dir = st.session_state.get('data_dir', '')
            _images_root = None
            if validate_directory(_data_dir):
                for _opt in _IMAGES_OPTIONS:
                    _cand = os.path.join(_data_dir, _opt)
                    if os.path.isdir(_cand):
                        _images_root = _cand
                        break

            _current_patient = st.session_state.get('patient', '')
            _pat_images_dir = None
            if _images_root and _current_patient:
                _cand = os.path.join(_images_root, _current_patient)
                if os.path.isdir(_cand):
                    _pat_images_dir = _cand

            _auto_proj_dir = None
            if _pat_images_dir:
                _img_fractions = sorted(
                    d for d in os.listdir(_pat_images_dir)
                    if os.path.isdir(os.path.join(_pat_images_dir, d))
                )
                if _img_fractions:
                    _plan_fx = st.session_state.get('fraction')
                    _default_fx = _plan_fx if _plan_fx in _img_fractions else _img_fractions[0]
                    _cur_treat_fx = st.session_state.get('treat_fraction', _default_fx)
                    if _cur_treat_fx not in _img_fractions:
                        _cur_treat_fx = _default_fx
                    st.selectbox('Fraction', _img_fractions,
                                 index=_img_fractions.index(_cur_treat_fx), key='treat_fraction',
                                 on_change=_on_treat_fraction_change)
                    _fx_dir = os.path.join(_pat_images_dir, st.session_state.get('fraction', _default_fx))

                    _imaging_types = [t for t in ['CBCT', 'KIM-KV'] if os.path.isdir(os.path.join(_fx_dir, t))]
                    if _imaging_types:
                        _default_type = 'CBCT' if 'CBCT' in _imaging_types else _imaging_types[0]
                        st.selectbox('Imaging type', _imaging_types,
                                     index=_imaging_types.index(_default_type), key='treat_imaging_type',
                                     on_change=_on_imaging_type_change)
                        _treat_type = st.session_state.get('treat_imaging_type', _default_type)
                        _type_dir = os.path.join(_fx_dir, _treat_type)

                        if _treat_type == 'CBCT':
                            _sessions = sorted(
                                d for d in os.listdir(_type_dir)
                                if os.path.isdir(os.path.join(_type_dir, d))
                            )
                            if _sessions:
                                st.selectbox('CBCT session', _sessions, key='cbct_session', on_change=_on_cbct_session_change)
                                _session_dir = os.path.join(_type_dir, st.session_state.get('cbct_session', _sessions[0]))
                                _found = _find_projection_files_recursive(_session_dir)
                                if _found:
                                    _auto_proj_dir = str(Path(_found[0]).parent)
                            else:
                                st.caption('No CBCT sessions found.')
                        else:
                            _found = _find_projection_files_recursive(_type_dir)
                            if _found:
                                _auto_proj_dir = str(Path(_found[0]).parent)
                    else:
                        st.caption('No CBCT or KIM-KV folders found in this fraction.')
                else:
                    st.caption(f'No fraction folders found under {_current_patient}.')
            elif _images_root:
                st.caption(f'No folder for patient "{_current_patient}" in Patient Images.')
            elif _data_dir:
                st.caption('No "Patient Images" folder found in workspace.')
            else:
                st.caption('Set a workspace folder to auto-detect treatment images.')

            if 'treatment_images' not in st.session_state:
                st.session_state.treatment_images = ''
            _ti_auto = st.session_state.get('_persisted', {}).get('treatment_images', {}).get('auto_select', True)
            if _auto_proj_dir and (not st.session_state.treatment_images or
                                   (_ti_auto and st.session_state.treatment_images != _auto_proj_dir)):
                st.session_state.treatment_images = _auto_proj_dir
                _mark_path_auto('treatment_images')

            st.text_input('Treatment images folder', key='treatment_images', on_change=_on_treatment_images_change)
            st.button('Browse treatment images folder', on_click=_browse_folder,
                      args=('treatment_images', 'Select treatment images folder'))
            _treat_files = infer_projection_files(st.session_state.get('treatment_images', ''))
            if _treat_files:
                _his_n = sum(1 for f in _treat_files if f.lower().endswith('.his'))
                _xim_n = sum(1 for f in _treat_files if f.lower().endswith('.xim'))
                _tiff_n = sum(1 for f in _treat_files if f.lower().endswith(('.tiff', '.tif')))
                if _his_n and not _xim_n and not _tiff_n:
                    _ext_lbl = '.his'
                elif _xim_n and not _his_n and not _tiff_n:
                    _ext_lbl = '.xim'
                elif _tiff_n and not _his_n and not _xim_n:
                    _ext_lbl = '.tiff'
                elif _his_n or _xim_n or _tiff_n:
                    _ext_lbl = '.xim/.his/.tiff'
                else:
                    _ext_lbl = 'projection'
                st.caption(f'Found {len(_treat_files)} ({_ext_lbl}) projection files.')
                if _tiff_n:
                    _tiff_angle_opts = ['kv-source', 'kv-detector', 'mv-source', 'mv-detector']
                    st.selectbox('Filename angle type', _tiff_angle_opts,
                                 key='tiff_filename_angle',
                                 help='The angle type encoded in each .tiff filename — used as the ground-truth angle for projection.')
            elif st.session_state.get('treatment_images'):
                st.caption('No projection files found in selected folder.')

            # Reload geometry when treatment_images changes.
            _curr_pdir = st.session_state.get('treatment_images', '')
            if st.session_state.get('_prev_treatment_images') != _curr_pdir:
                st.session_state['_prev_treatment_images'] = _curr_pdir
                if _curr_pdir:
                    try:
                        _geo_bar = st.progress(0, text='Loading projection geometry...')
                        _geo = load_geometry_from_projections(_curr_pdir, scale=st.session_state.get('proj_scale', 'IEC61217'))
                        _geo_bar.progress(100, text='Loading projection geometry...')
                        st.session_state['geometry_from_projections'] = _geo
                        for _k in ('sid', 'sdd', 'pixel_spacing', 'offset_x', 'matrix_width', 'matrix_height'):
                            if _k in _geo:
                                st.session_state[_k] = _geo[_k]
                        if 'matrix_width' in _geo and 'matrix_height' in _geo:
                            st.session_state['detector_size'] = _detector_size_label(
                                int(_geo['matrix_width']), int(_geo['matrix_height']))
                        st.session_state.pop('_geometry_error', None)
                    except Exception:
                        st.session_state['_geometry_error'] = traceback.format_exc()
                        st.rerun(scope='app')
                else:
                    st.session_state.pop('geometry_from_projections', None)


    _prospective = st.session_state.get('prospective_mode', False)
    _pdir = st.session_state.get('treatment_images', '') if not _prospective else ''
    _treat_frac = st.session_state.get('fraction')
    if _prospective:
        _treat_color = None  # Not shown in prospective mode.
    elif _pdir and infer_projection_files(_pdir):
        _treat_color = 'green'
    elif _treat_frac:
        _treat_color = 'red'
    else:
        _treat_color = 'orange'
    _geo_loaded = st.session_state.get('geometry_from_projections', {}).get('_loaded', set())
    _prosp_fields_set = st.session_state.get('_prosp_fields_set', set())

    def _gc(k: str) -> str:
        if _prospective:
            geo_k = 'detector_size' if k in ('matrix_width', 'matrix_height') else k
            return 'green' if geo_k in _prosp_fields_set else 'orange'
        geo_k = 'matrix_width' if k == 'detector_size' else k
        if geo_k in _geo_loaded:
            return 'green'
        return 'red' if _pdir else 'orange'

    # Trigger full app rerun when important state changes so log fragments update.
    # Must run AFTER all widgets above have rendered, so their keys are committed to
    # session state before we check for changes (otherwise st.rerun stops execution
    # before the widget renders and Streamlit cleans up its session state key).
    _sb_keys = ('data_dir', 'patient', 'fraction', 'planning_finalised',
                'treat_fraction', 'treat_imaging_type', 'cbct_session', 'treatment_images',
                'tiff_filename_angle', 'prospective_mode')
    _sb_prev = st.session_state.get('_sb_snapshot')
    _sb_curr = {k: st.session_state.get(k) for k in _sb_keys}
    if _sb_prev is None:
        st.session_state['_sb_snapshot'] = _sb_curr
    elif _sb_prev != _sb_curr:
        st.session_state['_sb_snapshot'] = _sb_curr
        _save_persisted_state()
        st.rerun(scope='app')

    _inject_field_styles([
        ('CT folder',                             _path_color(st.session_state.get('ct_dir', ''),       validate_directory)),
        ('RTSTRUCT file',                         _path_color(st.session_state.get('rtstruct_path', ''), validate_file)),
        ('RTPLAN file',                           _path_color(st.session_state.get('plan_path', ''),     validate_file)),
        ('Treatment images folder',               _treat_color),
        ('Detector pixel spacing (mm)',           _gc('pixel_spacing')),
        ('Source-to-isocentre distance (SID, mm)', _gc('sid')),
        ('Source-to-detector distance (SDD, mm)', _gc('sdd')),
        ('Detector offset X (mm)',                _gc('offset_x')),
        ('Detector offset Y (mm)',                _gc('offset_y')),
        ('Detector size',                         _gc('detector_size')),
    ])


with st.sidebar:
    render_sidebar()
    _render_proj_geometry()
    _render_image_settings()

if _geo_err := st.session_state.get('_geometry_error'):
    st.error('Failed to load projection geometry')
    st.code(_geo_err, language='python')

# Read widget values from session state (widgets live inside the sidebar fragment)
_do_create = st.session_state.pop('_create_triggered', False)
_n_proj        = int(st.session_state.get('n_proj', 8))
_angle_start   = float(st.session_state.get('angle_start', 0.0))
_angle_stop    = float(st.session_state.get('angle_stop', 359.0))
_method        = st.session_state.get('method', 'CTorch')
_det_size_label = st.session_state.get('detector_size', '1024x768')
_matrix_width, _matrix_height = _DETECTOR_SIZES.get(_det_size_label, (1024, 768))
_pixel_spacing = float(st.session_state.get('pixel_spacing', 0.388))
_sid           = float(st.session_state.get('sid', 1000.0))
_sdd           = float(st.session_state.get('sdd', 1500.0))
_offset_x      = float(st.session_state.get('offset_x', 0.0))
_offset_y      = float(st.session_state.get('offset_y', 0.0))

patients = []
root_for_patients = st.session_state.get('data_dir', '')
if validate_directory(st.session_state.get('data_dir', '')):
    for opt in _PLANS_OPTIONS:
        candidate = os.path.join(st.session_state.get('data_dir', ''), opt)
        if os.path.isdir(candidate):
            root_for_patients = candidate
            break
    patients = sorted([d for d in os.listdir(root_for_patients)
                       if os.path.isdir(os.path.join(root_for_patients, d))])


if _do_create:
    st.markdown("""<style>
section[data-testid="stSidebar"] { opacity: 0.4; pointer-events: none; }
</style>""", unsafe_allow_html=True)
    st.session_state.pop('planning_diagnostics', None)
    regions = st.session_state.get('loaded_regions', [])
    selected_region = st.session_state.get('selected_region') or _pick_default_region(regions)
    _projections_dir = st.session_state.get('treatment_images', '')
    projection_files = infer_projection_files(_projections_dir)
    _proj_ext = os.path.splitext(projection_files[0])[1].lower() if projection_files else ''
    _scale = st.session_state.get('proj_scale', 'IEC61217')
    _ordered_proj_files = None
    _frame_infos_by_file: 'dict[str, HisFrameInfo]' = {}
    if _proj_ext == '.his' and _projections_dir:
        try:
            _load_n = min(_n_proj, len(projection_files))
            _his_bar = st.progress(0, text='Loading .his files...')
            _angles_files, _frame_infos_by_file = load_his_angles_and_files(
                _projections_dir, angle_type='kv-detector', sort_by_angle=False,
                scale=_scale, n_angles=_n_proj)
            _his_bar.progress(100, text='Loading .his files...')
            angle_list = list(_angles_files.keys())
            if not angle_list:
                raise ValueError('No angles found in .his files')
            _ordered_proj_files = [os.path.join(_projections_dir, f) for f in _angles_files.values()]
            st.session_state['_angles_from_files'] = True
        except Exception:
            st.session_state['_angles_from_files'] = False
            angle_list = list(np.linspace(_angle_start, _angle_stop, _n_proj))
    elif _proj_ext == '.xim' and _projections_dir:
        try:
            _load_n = min(_n_proj, len(projection_files))
            _xim_bar = st.progress(0, text='Loading .xim files...')
            def _xim_progress(current: int, total: int) -> None:
                pct = int(current / total * 100) if total else 100
                _xim_bar.progress(pct, text=f'Loading .xim files... ({current}/{total})')
            _angles_files = load_xim_angles_and_files(
                _projections_dir, angle_type='kv-detector', sort_by_angle=False,
                scale=_scale, n_angles=_n_proj, progress_callback=_xim_progress)
            _xim_bar.progress(100, text='Loading .xim files...')
            angle_list = list(_angles_files.keys())
            if not angle_list:
                raise ValueError('No angles found in .xim files')
            _ordered_proj_files = [os.path.join(_projections_dir, f) for f in _angles_files.values()]
            st.session_state['_angles_from_files'] = True
        except Exception:
            st.session_state['_angles_from_files'] = False
            angle_list = list(np.linspace(_angle_start, _angle_stop, _n_proj))
    elif _proj_ext in ('.tiff', '.tif') and _projections_dir:
        try:
            _tiff_filename_angle = st.session_state.get('tiff_filename_angle', 'kv-source')
            _tiff_files = projection_files[:_n_proj]
            _tiff_bar = st.progress(0, text='Loading .tiff files...')
            # Load first file with data to infer machine type.
            _, _first_info = load_tiff(_tiff_files[0], filename_angle=_tiff_filename_angle)
            _machine = 'varian' if 'KVCollimatorX1' in _first_info else 'elekta'
            # Read angle metadata from every file without loading pixel data.
            _angles_files = {}
            for _t_idx, _t_f in enumerate(_tiff_files):
                _load_data = (_t_idx == 0)
                _, _t_info = load_tiff(_t_f, filename_angle=_tiff_filename_angle,
                                       machine=_machine, load_data=_load_data)
                _angles_files[_t_info['kv-detector-angle']] = os.path.basename(_t_f)
                _pct = int((_t_idx + 1) / len(_tiff_files) * 100)
                _tiff_bar.progress(_pct, text=f'Loading .tiff files... ({_t_idx + 1}/{len(_tiff_files)})')
            angle_list = list(_angles_files.keys())
            if not angle_list:
                raise ValueError('No angles found in .tiff files')
            _ordered_proj_files = [os.path.join(_projections_dir, f) for f in _angles_files.values()]
            st.session_state['_angles_from_files'] = True
        except Exception:
            st.session_state['_angles_from_files'] = False
            angle_list = list(np.linspace(_angle_start, _angle_stop, _n_proj))
    else:
        _parsed_angles = parse_projection_angles_from_filenames(projection_files) if projection_files else []
        if _parsed_angles:
            st.session_state['_angles_from_files'] = True
            angle_list = _parsed_angles if _n_proj >= len(_parsed_angles) else list(np.linspace(_parsed_angles[0], _parsed_angles[-1], _n_proj))
        else:
            st.session_state['_angles_from_files'] = False
            angle_list = list(np.linspace(_angle_start, _angle_stop, _n_proj))
    try:
        if not selected_region:
            raise ValueError('No region selected. Set RTSTRUCT path in the sidebar first.')

        ct_dir = st.session_state.get('ct_dir', '')
        rtstruct_path = st.session_state.get('rtstruct_path', '')

        ct_bar = st.progress(0, text='Loading CT...')

        def _ct_progress(current: int, total: int) -> None:
            pct = int(current / total * 100) if total else 100
            ct_bar.progress(pct, text=f'Loading CT... ({current}/{total} slices)')

        cache_bar = [None]

        def _on_cache_start() -> None:
            cache_bar[0] = st.progress(0, text='Caching CT...')

        def _on_cache_done() -> None:
            if cache_bar[0] is not None:
                cache_bar[0].progress(100, text='CT cached.')

        ct_volume, affine = load_ct(ct_dir, progress_callback=_ct_progress,
                                    on_cache_start=_on_cache_start, on_cache_done=_on_cache_done)
        ct_bar.progress(100, text='CT loaded.')

        isocentre = st.session_state.get('loaded_isocentre')
        if isocentre is None:
            spacing = affine_spacing(affine)
            shape = ct_volume.shape
            isocentre = np.array([spacing[i] * (shape[i] - 1) / 2 for i in range(3)], dtype=float)
            st.warning('No isocentre from RTPLAN — using CT volume centre.')

        rtstruct_bar = st.progress(0, text='Loading RTSTRUCT...')
        region_ids, region_labels = from_rtstruct_dicom(
            rtstruct_path, ct_volume.shape, affine,
            region_id=[selected_region],
            landmark_id=None,
        )
        label_batch = np.expand_dims(region_labels[0].astype(np.uint8), axis=0)
        rtstruct_bar.progress(100, text='RTSTRUCT loaded.')

        def _to_numpy(t):
            if hasattr(t, 'detach'):
                return t.detach().cpu().numpy()
            return np.asarray(t)

        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        proj_bar = st.progress(0, text=f'Creating projections (device={_device})...')

        def _proj_progress(current: int, total: int) -> None:
            pct = int(current / total * 100) if total else 100
            proj_bar.progress(pct, text=f'Creating projections (device={_device})... ({current}/{total})')

        if _method == 'CTorch':
            drrs, label_projs = project_ctorch(
                ct_volume, affine, isocentre, _sid, _sdd,
                (_matrix_width, _matrix_height), (_pixel_spacing, _pixel_spacing),
                (_offset_x, _offset_y), angle_list, labels=label_batch,
                progress_callback=_proj_progress,
            )
        elif _method == 'DiffDRR':
            drrs, label_projs = project_diffdrr(
                ct_volume, affine, isocentre, _sid, _sdd,
                (_matrix_width, _matrix_height), (_pixel_spacing, _pixel_spacing),
                (_offset_x, _offset_y), angle_list, labels=label_batch,
                progress_callback=_proj_progress,
            )
        else:
            drrs, label_projs = project_igt(
                ct_volume, affine, isocentre, _sid, _sdd,
                (_matrix_width, _matrix_height), (_pixel_spacing, _pixel_spacing),
                (_offset_x, _offset_y), angle_list, labels=label_batch,
                progress_callback=_proj_progress,
            )
        proj_bar.progress(100, text='Projections created.')
        st.session_state.drr_result = {
            'drrs': _to_numpy(drrs).astype(np.float32),
            'labels': _to_numpy(label_projs).astype(bool),
            'angles': angle_list,
            'region': selected_region,
            'method': _method,
            'proj_files': _ordered_proj_files,
            'frame_infos': _frame_infos_by_file,
        }
        st.session_state.current_proj_idx = 1
        st.session_state.contour_offset_x = 0
        st.session_state.contour_offset_y = 0
        if st.session_state.get('prospective_mode', False):
            _compute_onscreen_status()
        for k in list(st.session_state.keys()):
            if k.startswith('_undo_stack_'):
                del st.session_state[k]
        # Auto-load contour offsets if a saved file already exists (retrospective only).
        if not st.session_state.get('prospective_mode', False):
            _auto_csv = _get_offsets_save_path()
            if Path(_auto_csv).is_file():
                _load_contour_offsets(_auto_csv)
        _save_persisted_state(log_key='_state_saved_projections')
        # Rerun immediately to clear all progress bars and show clean results
        st.rerun()
    except Exception as exc:
        tb = traceback.format_exc()
        st.error(f'Failed to create projections: {exc}')
        st.code(tb, language='python')

elif 'drr_result' in st.session_state:
    render_drr_display()

else:
    _ct  = st.session_state.get('ct_dir', '')
    _rs  = st.session_state.get('rtstruct_path', '')
    _rp  = st.session_state.get('plan_path', '')
    _pat = st.session_state.get('patient', '')
    _paths_ok = validate_directory(_ct) and validate_file(_rs) and validate_file(_rp)
    if st.session_state.get('data_dir') and not _pat:
        st.info('Select a patient in the sidebar.')
    elif _pat and not _paths_ok:
        st.info('Fix missing input paths in the sidebar, then click "Finalise planning data".')
    elif _pat and not st.session_state.get('planning_finalised'):
        st.info('Click "Finalise planning data" in the sidebar to continue.')
    elif st.session_state.get('planning_finalised'):
        st.info('Configure treatment images and geometry, then click "Create projections".')

_log_drr_section(patients, root_for_patients)
