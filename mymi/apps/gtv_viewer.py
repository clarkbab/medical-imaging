from dicomset.dicom.utils import from_ct_dicom, from_rtstruct_dicom
from dicomset.utils import affine_spacing, centre_of_mass, foreground_fov, foreground_fov_centre, volume
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import re
import os

DATA_DIR = r"D:\Brett\data\mymi\files\mlm\valkim\gt"


def phase_label(folder_name):
    m = re.search(r'(\d+)', folder_name)
    return m.group(1) if m else folder_name


def get_view_aspect(view, affine):
    spacing = affine_spacing(affine)
    if view == 0:
        aspect = spacing[2] / spacing[1]
    elif view == 1:
        aspect = spacing[2] / spacing[0]
    elif view == 2:
        aspect = spacing[1] / spacing[0]
    return np.abs(aspect)


def get_view_origin(view, orientation='LPS'):
    if view == 0:
        origin_x = 'lower' if orientation[1] == 'P' else 'upper'
        origin_y = 'lower' if orientation[2] == 'S' else 'upper'
    elif view == 1:
        origin_x = 'lower' if orientation[0] == 'L' else 'upper'
        origin_y = 'lower' if orientation[2] == 'S' else 'upper'
    else:
        origin_x = 'lower' if orientation[0] == 'L' else 'upper'
        origin_y = 'upper' if orientation[1] == 'P' else 'lower'
    return (origin_x, origin_y)


def get_view_slice(view, data, idx):
    n_dims = len(data.shape)
    if n_dims == 4:
        assert data.shape[0] == 3
        view_idx = view + 1
    else:
        view_idx = view

    if idx >= data.shape[view_idx]:
        raise ValueError(f"Idx '{idx}' out of bounds, only '{data.shape[view_idx]}' slices.")

    data_index = [slice(None)] if n_dims == 4 else []
    for i in range(3):
        v_idx = i + 1 if n_dims == 4 else i
        data_index += [idx if i == view else slice(data.shape[v_idx])]
    data_index = tuple(data_index)
    slice_data = data[data_index]

    slice_data = np.transpose(slice_data)
    return slice_data, idx


def get_crop_box(centre_voxel, view, spacing, crop_mm, img_shape):
    if view == 0:
        row_c, col_c = centre_voxel[2], centre_voxel[1]
        row_sp, col_sp = abs(spacing[2]), abs(spacing[1])
    elif view == 1:
        row_c, col_c = centre_voxel[2], centre_voxel[0]
        row_sp, col_sp = abs(spacing[2]), abs(spacing[0])
    else:
        row_c, col_c = centre_voxel[1], centre_voxel[0]
        row_sp, col_sp = abs(spacing[1]), abs(spacing[0])
    half_r = int(crop_mm / (2 * row_sp))
    half_c = int(crop_mm / (2 * col_sp))
    r0 = max(0, row_c - half_r)
    r1 = min(img_shape[0], row_c + half_r)
    c0 = max(0, col_c - half_c)
    c1 = min(img_shape[1], col_c + half_c)
    return r0, r1, c0, c1


N_COLS = 5
VIEW_MAP = {"Axial": 2, "Sagittal": 0, "Coronal": 1}

st.set_page_config(layout="wide", page_title="GTV Contour Viewer")
st.title("GTV Contour Viewer")

# Arrow key support — listens for left/right and clicks the ◀/▶ buttons.
components.html("""
<script>
const doc = window.parent.document;
doc.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') {
        const btn = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim() === '◀');
        if (btn) btn.click();
    } else if (e.key === 'ArrowRight') {
        const btn = Array.from(doc.querySelectorAll('button')).find(b => b.innerText.trim() === '▶');
        if (btn) btn.click();
    }
});
</script>
""", height=0)


@st.cache_resource(show_spinner=False)
def _load_phase(data_dir, patient, phase):
    """Load CT/RTSTRUCT and compute metrics for a single phase."""
    phase_dir = os.path.join(data_dir, patient, phase)
    ct_dir = os.path.join(phase_dir, "ct")
    rtstruct_path = os.path.join(phase_dir, "rtstruct.dcm")
    if not os.path.isdir(ct_dir) or not os.path.isfile(rtstruct_path):
        return None

    ct_data, affine = from_ct_dicom(ct_dir)
    label_data, regions = from_rtstruct_dicom(rtstruct_path, ct_data.shape, affine, region_id='all', return_regions=True)

    volumes = []
    centroids = []
    for ri, region in enumerate(regions):
        vol_mm3 = volume(label_data[ri], affine=affine)
        vol_cc = vol_mm3 / 1000.0
        volumes.append({'Phase': phase, 'Region': region, 'Volume (cc)': vol_cc})
        com = centre_of_mass(label_data[ri], affine=affine)
        centroids.append({'Phase': phase, 'Region': region, 'X (mm)': com[0], 'Y (mm)': com[1], 'Z (mm)': com[2]})

    return {
        'ct': ct_data,
        'affine': affine,
        'labels': label_data,
        'regions': regions,
        'volumes': volumes,
        'centroids': centroids,
    }


def load_patient_data(patient, data_dir, selected_phases=None):
    patient_dir = os.path.join(data_dir, patient)
    phases = sorted([d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))])
    if selected_phases is not None:
        phases = [p for p in phases if p in selected_phases]

    phase_data = {}
    all_regions = []
    all_volumes = []
    all_centroids = []

    progress = st.progress(0)
    for i, phase in enumerate(phases):
        result = _load_phase(data_dir, patient, phase)
        if result is not None:
            phase_data[phase] = {
                'ct': result['ct'],
                'affine': result['affine'],
                'labels': result['labels'],
                'regions': result['regions'],
            }
            all_volumes.extend(result['volumes'])
            all_centroids.extend(result['centroids'])
            for r in result['regions']:
                if r not in all_regions:
                    all_regions.append(r)
        progress.progress((i + 1) / len(phases))
    progress.empty()

    # Precompute union FOV across all regions in all phases.
    fov_min = np.full(3, np.inf)
    fov_max = np.full(3, -np.inf)
    for pdata in phase_data.values():
        for ri in range(len(pdata['regions'])):
            fov = foreground_fov(pdata['labels'][ri])
            if fov is not None:
                fov_min = np.minimum(fov_min, fov[0])
                fov_max = np.maximum(fov_max, fov[1])
    union_fov = (fov_min, fov_max)

    volumes_df = pd.DataFrame(all_volumes)
    centroids_df = pd.DataFrame(all_centroids)

    return phase_data, all_regions, union_fov, volumes_df, centroids_df


# --- Sidebar: patient selection ---
def _browse_folder():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    folder = filedialog.askdirectory(initialdir=st.session_state.get('data_dir', DATA_DIR))
    root.destroy()
    if folder:
        st.session_state.data_dir = folder

if 'data_dir' not in st.session_state:
    st.session_state.data_dir = DATA_DIR

with st.sidebar:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        data_dir = st.text_input("Data folder", key="data_dir")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("📁", on_click=_browse_folder)
    if not os.path.isdir(data_dir):
        st.error("Folder not found.")
        st.stop()
    patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not patients:
        st.warning("No patient folders found.")
        st.stop()
    def _on_patient_change():
        st.session_state.pop('loaded_data', None)
    patient = st.selectbox("Patient", patients, on_change=_on_patient_change)

    # Phase filter checkboxes.
    patient_dir = os.path.join(data_dir, patient)
    available_phases = sorted([d for d in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, d))])
    def _toggle_all_phases():
        val = st.session_state.phase_all
        for p in available_phases:
            st.session_state[f"phase_{p}"] = val
    st.checkbox("All", value=True, key="phase_all", on_change=_toggle_all_phases)
    n_phase_cols = min(len(available_phases), 5) or 1
    phase_cols = st.columns(n_phase_cols)
    selected_phases = []
    for i, phase in enumerate(available_phases):
        with phase_cols[i % n_phase_cols]:
            if st.checkbox(phase_label(phase), value=True, key=f"phase_{phase}"):
                selected_phases.append(phase)

    if st.button("Visualise", type="primary"):
        _load_phase.clear()
        result = load_patient_data(patient, data_dir, tuple(selected_phases))
        st.session_state.current_patient = patient
        st.session_state.current_data_dir = data_dir
        st.session_state.selected_phases = tuple(selected_phases)
        st.session_state.loaded_data = result
        # Reset slice position for the new patient.
        st.session_state.pop('slice_idx', None)
        st.session_state.pop('_view_centre_key', None)

if 'loaded_data' not in st.session_state:
    st.info("Select a patient and click **Visualise** to load data.")
    st.stop()

# --- Load data (from session state) ---
phase_data, all_regions, union_fov, volumes_df, centroids_df = st.session_state.loaded_data

phases = list(phase_data.keys())
if not phases:
    st.error("No valid phase data found for this patient.")
    st.stop()

first_data = phase_data[phases[0]]

# --- Sidebar: view / region / crop controls ---
with st.sidebar:
    view_name = st.selectbox("View", list(VIEW_MAP.keys()))
    view = VIEW_MAP[view_name]

    centre_region = st.selectbox("Centre Region", all_regions, index=0)
    crop_mm = st.number_input("Crop FOV (mm)", value=100, min_value=10, max_value=500, step=10)

    window_preset = st.selectbox("CT Window", ["Lung (1500, -600)", "Tissue (400, 40)", "Bone (1800, 400)", "Auto"])

    st.markdown("---")
    st.subheader("Regions")
    st.markdown("""
        <style>
        [data-testid="stSidebar"] .stCheckbox { margin-bottom: -15px; }
        [data-testid="stSidebar"] .stMarkdown { margin-bottom: -15px; }
        </style>
    """, unsafe_allow_html=True)
    region_colours = sns.color_palette('colorblind', n_colors=len(all_regions))
    visible_regions = {}
    for i, r in enumerate(all_regions):
        hex_c = mpl.colors.to_hex(region_colours[i])
        c1, c2 = st.columns([4, 1])
        with c1:
            visible_regions[r] = st.checkbox(r, value=True, key=f"vis_{r}")
        with c2:
            st.markdown(f'<span style="color:{hex_c}; font-size:1.4em; line-height:2.4">■</span>', unsafe_allow_html=True)

# Clamp union FOV to volume bounds with padding.
SLICE_PAD = 5
fov_min, fov_max = union_fov
ct_shape = first_data['ct'].shape
slice_lo = [max(0, int(fov_min[ax]) - SLICE_PAD) for ax in range(3)]
slice_hi = [min(ct_shape[ax] - 1, int(fov_max[ax]) + SLICE_PAD) for ax in range(3)]

# --- Compute centre and initial slice from first phase's selected region ---
if centre_region in first_data['regions']:
    ridx = first_data['regions'].index(centre_region)
    centre = foreground_fov_centre(first_data['labels'][ridx])
    region_fov = foreground_fov(first_data['labels'][ridx])
    if centre is not None:
        centre_voxel = tuple(int(c) for c in centre)
    else:
        centre_voxel = tuple(s // 2 for s in ct_shape)
    if region_fov is not None:
        initial_slice = int(region_fov[0][view]) - 1
    else:
        initial_slice = int(centre_voxel[view])
else:
    centre_voxel = tuple(s // 2 for s in ct_shape)
    initial_slice = int(centre_voxel[view])

# Reset slice index when view or centre region changes.
view_centre_key = f"{view}_{centre_region}"
min_slice = slice_lo[view]
max_slice = slice_hi[view]
if st.session_state.get('_view_centre_key') != view_centre_key:
    st.session_state['_view_centre_key'] = view_centre_key
    st.session_state['slice_idx'] = np.clip(initial_slice, min_slice, max_slice)

@st.fragment
def render_slice_viewer():
    # --- Slice navigation ---
    def prev_slice():
        st.session_state.slice_idx = max(min_slice, st.session_state.slice_idx - 1)

    def next_slice():
        st.session_state.slice_idx = min(max_slice, st.session_state.slice_idx + 1)

    col_min, col_prev, col_slider, col_next, col_max = st.columns([1, 1, 8, 1, 1])
    with col_min:
        st.markdown(f"**{min_slice}**")
    with col_prev:
        st.button("◀", on_click=prev_slice)
    with col_slider:
        slice_idx = st.slider("Slice", min_slice, max_slice, key='slice_idx', label_visibility='collapsed')
    with col_next:
        st.button("▶", on_click=next_slice)
    with col_max:
        st.markdown(f"**{max_slice}**")

    # --- CT window/level ---
    window_settings = {
        "Tissue (400, 40)": (400, 40),
        "Bone (1800, 400)": (1800, 400),
        "Lung (1500, -600)": (1500, -600),
        "Auto": None,
    }
    window = window_settings[window_preset]

    # --- Colour palette ---
    colours = sns.color_palette('colorblind', n_colors=len(all_regions))

    # --- Render phase grid ---
    n_phases = len(phases)
    n_rows = (n_phases + N_COLS - 1) // N_COLS

    for row_idx in range(n_rows):
        cols = st.columns(N_COLS)
        for col_idx in range(N_COLS):
            phase_idx = row_idx * N_COLS + col_idx
            if phase_idx >= n_phases:
                break

            phase = phases[phase_idx]
            pdata = phase_data[phase]
            ct_data = pdata['ct']
            affine = pdata['affine']
            label_data = pdata['labels']
            phase_regions = pdata['regions']

            with cols[col_idx]:
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))

                phase_max = ct_data.shape[view] - 1
                s_idx = min(slice_idx, phase_max)

                ct_slice, _ = get_view_slice(view, ct_data, s_idx)
                aspect = get_view_aspect(view, affine)
                origin_y = get_view_origin(view)[1]
                spacing = affine_spacing(affine)
                r0, r1, c0, c1 = get_crop_box(centre_voxel, view, spacing, crop_mm, ct_slice.shape)
                ct_crop = ct_slice[r0:r1, c0:c1]

                if window is not None:
                    width, level = window
                    vmin = level - width / 2
                    vmax = level + width / 2
                else:
                    vmin, vmax = float(ct_crop.min()), float(ct_crop.max())

                ax.imshow(ct_crop, cmap='gray', aspect=aspect, origin=origin_y, vmin=vmin, vmax=vmax)

                for region in all_regions:
                    if not visible_regions.get(region, True):
                        continue
                    if region not in phase_regions:
                        continue
                    ri = phase_regions.index(region)
                    ci = all_regions.index(region)
                    colour = colours[ci]

                    label_slice, _ = get_view_slice(view, label_data[ri], s_idx)
                    label_crop = label_slice[r0:r1, c0:c1]
                    if label_crop.max() == 0:
                        continue
                    cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), colour))
                    ax.imshow(label_crop, alpha=0.3, aspect=aspect, cmap=cmap, interpolation='none', origin=origin_y)
                    ax.contour(label_crop, colors=[colour], levels=[.5], linestyles='solid')

                ax.set_title(phase_label(phase), fontsize=10)
                ax.axis('off')
                fig.tight_layout(pad=0.5)
                st.pyplot(fig)
                plt.close(fig)

render_slice_viewer()

# --- Metric plots (rendered with visible region filter) ---
st.markdown("---")
phase_names = sorted(phase_data.keys())
plot_phase_labels = [phase_label(p) for p in phase_names]
plot_colours = sns.color_palette('colorblind', n_colors=len(all_regions))
plot_regions = [r for r in all_regions if visible_regions.get(r, True)]

# Volume line plot.
st.subheader("Region Volumes")
fig_vol, ax_vol = plt.subplots(figsize=(max(12, len(phase_names) * 2), 3))
for region in plot_regions:
    ci = all_regions.index(region)
    region_vols = volumes_df[volumes_df['Region'] == region]
    vals = [region_vols[region_vols['Phase'] == p]['Volume (cc)'].values[0]
            if p in region_vols['Phase'].values else np.nan
            for p in phase_names]
    ax_vol.plot(plot_phase_labels, vals, marker='o', label=region, color=plot_colours[ci])
ax_vol.set_xlabel('Phase')
ax_vol.set_ylabel('Volume (cc)')
if plot_regions:
    ax_vol.legend()
fig_vol.tight_layout()
st.pyplot(fig_vol)
plt.close(fig_vol)

# Centroid position plot.
st.subheader("Centroid Position")
fig_com, axes_com = plt.subplots(1, 3, figsize=(max(12, len(phase_names) * 2), 3))
axis_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']
for ai, axis_label in enumerate(axis_labels):
    ax = axes_com[ai]
    for region in plot_regions:
        ci = all_regions.index(region)
        region_coms = centroids_df[centroids_df['Region'] == region]
        vals = [region_coms[region_coms['Phase'] == p][axis_label].values[0]
                if p in region_coms['Phase'].values else np.nan
                for p in phase_names]
        ax.plot(plot_phase_labels, vals, marker='o', label=region, color=plot_colours[ci])
    ax.set_xlabel('Phase')
    ax.set_ylabel(axis_label)
    ax.set_title(axis_label)
# Set all subplots to the same y-range (centered on each axis's midpoint).
axis_ranges = []
for ax in axes_com:
    ymin, ymax = ax.get_ylim()
    axis_ranges.append(ymax - ymin)
max_range = max(axis_ranges) if axis_ranges else 1
for ax in axes_com:
    ymin, ymax = ax.get_ylim()
    mid = (ymin + ymax) / 2
    ax.set_ylim(mid - max_range / 2, mid + max_range / 2)
if plot_regions:
    axes_com[-1].legend()
fig_com.tight_layout()
st.pyplot(fig_com)
plt.close(fig_com)
