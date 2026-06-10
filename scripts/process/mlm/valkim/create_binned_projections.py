import dicomset as ds
from dicomset.utils import arg_to_list
from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.utils import save_numpy
import numpy as np
import os
from tqdm import tqdm

from mymi.utils.cdog import load_tiff_arc, list_tiff_arcs, list_tiff_fractions, to_4dct_phase
from mymi.utils.projections import project_ctorch

# This creates the GT data for model evaluation.
# Doesn't interpolate between 4DCT phases though.

dataset = 'VALKIM-PP'
inh_series = 'series_0'
exh_series = 'series_5'
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

    # Get the fractions for this patient.
    pat_id_valkim = pat_ids_valkim[p]
    patpath = rf"R:\2RESEARCH\1_ClinicalData\VALKIM\RNSH\Treatment files\{pat_id_valkim}"
    pat_fractions = arg_to_list(fractions, int, literals={'all': list_tiff_fractions(patpath)})
    print(f'Found fractions: {pat_fractions}')

    for f in tqdm(pat_fractions, leave=False):
        fracpath = os.path.join(patpath, f"Fx{f:02d}")
        print(fracpath)

        # Get the arcs for this fraction.
        pat_arcs = arg_to_list(arcs, int, literals={'all': list_tiff_arcs(fracpath)})
        print(f'Found arcs: {pat_arcs}')

        for a in tqdm(pat_arcs, leave=False):
            # Load the arc data.
            data, info = load_tiff_arc(fracpath, arc=a, filename_angle=filename_angle)
            kv_det_angles = info['kv-detector-angle'].values
            breathing_phases = info['MMPhase0'].values
            phases_4dct = [to_4dct_phase(p) for p in breathing_phases]

            # Bin everything by 4DCT phase - we need to project in groups.
            # Map from 4DCT phase to treatment image frames.
            n_phases = 10
            phases = list(range(n_phases))
            phase_frames = [[i for i, (_, (tp, _)) in enumerate(zip(kv_det_angles, phases_4dct)) if tp == p] for p in phases]

            # Create placeholder for projection images and labels.
            n_angles = len(kv_det_angles)
            det_size = info['det-size'].values[0]
            ct_proj = np.zeros((n_angles, *det_size))
            n_label_channels = 2
            labels_proj = np.zeros((n_angles, n_label_channels, *det_size))

            # Create projections by phase.
            for ph in tqdm(phases, desc='Projecting phases', leave=False):
                # Load phase data.
                # Inh/exh phases have real GTVs, intermediate phases have corrfield-propagated GTV.
                regions = ['GTV', 'ts_Lung'] if ph in (0, 5) else ['cf_GTV', 'ts_Lung']
                assert len(regions) == n_label_channels
                phase_series = f'series_{ph}'
                phase_ct = pat.ct_series(phase_series).data
                phase_affine = pat.ct_series(phase_series).affine
                phase_regions, phase_labels = pat.regions_series(phase_series).data(r=regions)

                # Check that all images share projection geometry.
                frames = phase_frames[ph]
                first_info = info.iloc[frames[0]]
                sid = first_info['sid']
                sdd = first_info['sdd']
                det_size = first_info['det-size']
                det_spacing = first_info['det-spacing']
                det_offset = first_info['det-offset']
                print(treatment_iso, sid, sdd, det_size, det_spacing, det_offset)
                for i in frames[1:]:
                    other_info = info.iloc[i]
                    osid = other_info['sid']
                    osdd = other_info['sdd']
                    odet_size = other_info['det-size']
                    odet_spacing = other_info['det-spacing']
                    odet_offset = other_info['det-offset']
                    assert osid == sid, f"SID mismatch: {osid} vs {sid}"
                    assert osdd == sdd, f"SDD mismatch: {osdd} vs {sdd}"
                    assert np.allclose(odet_size, det_size), f"Detector size mismatch: {odet_size} vs {det_size}"
                    assert np.allclose(odet_spacing, det_spacing), f"Detector spacing mismatch: {odet_spacing} vs {det_spacing}"
                    assert np.allclose(odet_offset, det_offset), f"Detector offset mismatch: {odet_offset} vs {det_offset}"
                print('Geometry matched!')
                
                phase_angles = [kv_det_angles[i] for i in frames]
                ct_proj_phase, labels_proj_phase = project_ctorch(phase_ct, phase_affine, treatment_iso,
                    sid, sdd, det_size, det_spacing, det_offset, phase_angles, labels=phase_labels)
                ct_proj[frames] = ct_proj_phase
                labels_proj[frames] = labels_proj_phase

            dirpath = os.path.join(set.path, 'data', 'projections', pat.id, f'Fx{f:02d}')
            os.makedirs(dirpath, exist_ok=True)
            save_numpy(data, os.path.join(dirpath, 'treatment_binned.npz'))
            save_numpy(ct_proj, os.path.join(dirpath, 'ct_proj_binned.npz'))
            save_numpy(labels_proj, os.path.join(dirpath, 'labels_proj_binned.npz'))
