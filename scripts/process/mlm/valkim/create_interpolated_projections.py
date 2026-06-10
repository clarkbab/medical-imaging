import dicomset as ds
from dicomset.utils import arg_to_list
from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.utils import save_numpy
import numpy as np
import os
from tqdm import tqdm

from mymi.processing.interpolation import interpolate_masks_radial_3d
from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions, to_4dct_phase
from mymi.utils.projections import project_ctorch

# Creates CT and label projections for each treatment frame by radially
# interpolating labels between the two bounding 4DCT phases, then projecting
# the current-phase CT volume with the interpolated labels.

dataset = 'VALKIM-PP'
set = ds.load(dataset, 'nifti')
pat_ids = ['PAT1', 'PAT2', 'PAT3', 'PAT4']
pat_ids = ['PAT1']
fractions = 'all'
fractions = [1]
arcs = 'all'
arcs = [1]
filename_angles = ['kv-source', 'kv-source', '?', 'mv-source']
pat_ids_valkim = [f'Patient{int(p.replace("PAT", "")):02d}' for p in pat_ids]

for p in tqdm(range(len(pat_ids))):
    pat_id = pat_ids[p]
    pat = set.patient(pat_id)
    filename_angle = filename_angles[p]
    print(pat)

    # Load planned iso.
    rtplan = pat.dicom.default_rtplan.dicom
    rtplan_info = from_rtplan_dicom(rtplan)
    treatment_iso = rtplan_info['isocentre']

    pat_id_valkim = pat_ids_valkim[p]
    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\{pat_id_valkim}"
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
            kv_det_angles = info['kv-detector-angle'].values
            breathing_phases = info['MMPhase0'].values
            phases_4dct = [to_4dct_phase(p) for p in breathing_phases]

            # Create placeholders for projection images and labels.
            n_angles = len(kv_det_angles)
            det_size = info.iloc[0]['det-size']
            ct_proj = np.zeros((n_angles, *det_size))
            n_label_channels = 2
            labels_proj = np.zeros((n_angles, n_label_channels, *det_size))

            # Project each treatment frame with phase-interpolated labels.
            for i in tqdm(range(n_angles), desc='Projecting frames', leave=False):
                phase_4dct, phase_4dct_di = phases_4dct[i]

                # Load current phase CT and labels.
                # Inh/exh phases have real GTVs; intermediate phases have corrfield-propagated GTVs.
                regions = ['GTV', 'ts_Lung'] if phase_4dct in (0, 5) else ['cf_GTV', 'ts_Lung']
                assert len(regions) == n_label_channels
                phase_series = f'series_{phase_4dct}'
                phase_ct = pat.ct_series(phase_series).data
                phase_affine = pat.ct_series(phase_series).affine
                phase_labels = pat.regions_series(phase_series).data(r=regions, rr=False)

                # Load next phase labels for interpolation.
                next_phase_4dct = (phase_4dct + 1) % 10
                next_phase_series = f'series_{next_phase_4dct}'
                next_phase_labels = pat.regions_series(next_phase_series).data(r=regions, rr=False)

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

                ct_proj_i, labels_proj_i = project_ctorch(phase_ct, phase_affine, treatment_iso,
                    sid, sdd, det_size, det_spacing, det_offset, [kv_det_angles[i]],
                    labels=interp_labels)
                ct_proj[[i]] = ct_proj_i
                labels_proj[[i]] = labels_proj_i

            dirpath = os.path.join(set.path, 'data', 'projections', pat.id, f'Fx{f:02d}')
            os.makedirs(dirpath, exist_ok=True)
            save_numpy(data, os.path.join(dirpath, 'treatment_interp.npz'))
            save_numpy(ct_proj, os.path.join(dirpath, 'ct_proj_interp.npz'))
            save_numpy(labels_proj, os.path.join(dirpath, 'labels_proj_interp.npz'))
