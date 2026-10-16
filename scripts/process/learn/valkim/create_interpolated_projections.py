import dicomset as ds
from dicomset.utils import arg_to_list, logger, save_numpy, save_csv, from_rtplan_dicom
from itertools import islice
import numpy as np
import os
from tqdm import tqdm

from mymi.processing.interpolation import interpolate_masks_radial_3d
from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions, to_4dct_phase
from mymi.utils.projections import project_ctorch, project_igt


# Creates CT and label projections for each treatment frame by radially
# interpolating labels between the two bounding 4DCT phases, then projecting
# the current-phase CT volume with the interpolated labels.

dataset = 'VALKIM-PP'
set = ds.load(dataset, 'nifti')
pat_ids = [1, 2, 3, 4]
pat_ids = [1]
fractions = 'all'
# fractions = [1]
arcs = 'all'
# arcs = [1]
n_frames = None
# n_frames = 5
blend_ct = True
labels_output = 'density'
filename_angles = {1: 'kv-source', 2: 'kv-source', 3: '?', 4: 'mv-source'}
n_phases = 10
n_label_channels = 5

def get_phase_regions(phase: int) -> list:
    return ['GTV' if phase in (0, 5) else 'cf_GTV', 'bc_Marker_1', 'bc_Marker_2', 'bc_Marker_3', 'ts_Lung']

for p in tqdm(pat_ids):
    pat = set.patient(f'PAT{p}')
    filename_angle = filename_angles[p]

    # Load planned iso.
    rtplan = pat.dicom.default_rtplan.dicom
    rtplan_info = from_rtplan_dicom(rtplan)
    treatment_iso = rtplan_info['isocentre']

    # Load all 4DCT phases (CT and labels) once per patient.
    # Inh/exh phases have real GTVs; intermediate phases have corrfield-propagated GTVs.
    phase_data = {}
    for ph in tqdm(range(n_phases), desc='Loading phases', leave=False):
        regions = get_phase_regions(ph)
        series = pat.ct_series(f'series_{ph}')
        loaded_regions, labels = pat.regions_series(f'series_{ph}').data(r=regions, rr=True, sort_regions=False)
        assert loaded_regions == regions
        phase_data[ph] = dict(ct=series.data, affine=series.affine, labels=labels)

    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\Patient{p:02d}"
    pat_fractions = arg_to_list(fractions, int, literals={'all': list_tiff_fractions(patpath)})
    print(f'Found fractions: {pat_fractions}')

    for f in tqdm(pat_fractions, leave=False):
        fracpath = os.path.join(patpath, f"Fx{f:02d}")
        print(fracpath)

        pat_arcs = arg_to_list(arcs, int, literals={'all': list_tiff_arcs(fracpath)})
        print(f'Found arcs: {pat_arcs}')

        for a in tqdm(pat_arcs, leave=False):
            # Load arc data.
            data, info = load_tiff_arc(fracpath, arc=a, filename_angle=filename_angle)
            kv_source_angles = info['kv-source-angle'].values
            breathing_phases = info['MMPhase0'].values
            phases_4dct = [to_4dct_phase(p) for p in breathing_phases]

            # Create placeholders for projection images and labels.
            n_angles = len(kv_source_angles) if n_frames is None else min(len(kv_source_angles), n_frames)
            det_size = info.iloc[0]['det-size']
            ct_proj = np.zeros((n_angles, *det_size))
            labels_proj = np.zeros((n_angles, n_label_channels, *det_size))

            # Project each treatment frame with phase-interpolated labels.
            for i in tqdm(range(n_angles), desc='Projecting frames', leave=False, total=n_angles):
                phase_4dct, phase_4dct_di = phases_4dct[i]

                # Get current and next phase CT and labels from the preloaded phases.
                next_phase_4dct = (phase_4dct + 1) % n_phases
                phase_ct = phase_data[phase_4dct]['ct']
                phase_affine = phase_data[phase_4dct]['affine']
                phase_labels = phase_data[phase_4dct]['labels']
                next_phase_labels = phase_data[next_phase_4dct]['labels']

                # Optionally blend CT intensities between phase x and x+1.
                if blend_ct:
                    next_phase_ct = phase_data[next_phase_4dct]['ct']
                    proj_ct = (1 - phase_4dct_di) * phase_ct + phase_4dct_di * next_phase_ct
                else:
                    proj_ct = phase_ct

                # Radially interpolate each label channel between current and next phase.
                interp_labels = np.stack([
                    interpolate_masks_radial_3d(la, lb, phase_4dct_di)
                    for la, lb in zip(phase_labels, next_phase_labels)
                ], axis=0)

                # Read per-frame projection geometry.
                frame_info = info.iloc[i]
                sid = frame_info['sid']
                sdd = frame_info['sdd']
                det_size = frame_info['det-size']
                det_spacing = frame_info['det-spacing']
                det_offset = frame_info['det-offset']
                print(treatment_iso, sid, sdd, det_size, det_spacing, det_offset)

                ct_proj_i, labels_proj_i = project_igt(proj_ct, phase_affine, treatment_iso,
                    sid, sdd, det_size, det_spacing, det_offset, [kv_source_angles[i]],
                    labels=interp_labels, labels_output=labels_output)
                ct_proj[[i]] = ct_proj_i
                labels_proj[[i]] = labels_proj_i

            logger.info(f'Saving projections for PAT{p} Fx{f:02d} Arc{a:02d}')
            dirpath = os.path.join(set.path, 'data', 'projections', pat.id, f'Fx{f:02d}', f'Arc{a:02d}')
            os.makedirs(dirpath, exist_ok=True)
            save_numpy(kv_source_angles, os.path.join(dirpath, 'kv_source_angles_interp.npz'))
            save_numpy(data, os.path.join(dirpath, 'treatment_interp.npz'))
            save_csv(info, os.path.join(dirpath, 'treatment_info_interp.csv'))
            save_numpy(ct_proj, os.path.join(dirpath, 'ct_proj_interp.npz'))
            label_name = 'labels_proj_interp.npz' if labels_output == 'boolean' else 'labels_proj_interp_raw.npz'
            save_numpy(labels_proj, os.path.join(dirpath, label_name))
