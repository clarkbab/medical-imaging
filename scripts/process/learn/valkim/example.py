from dicomset.dicom.utils import from_rtplan_dicom
from dicomset.utils import save_numpy
from mymi.processing import project_ctorch
from mymi.predictions.registration import interpolate_masks_radial_3d

def treatment_phase_to_4dct(
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

# Load arc data.
arc_data, arc_info = load_tiff_arc(dirpath, 1)
treatment_angles = [i['kv-source-angle'] for i in arc_info]
treatment_phases = [i['MMPhase0'] for i in arc_info]
phases_4dct, phases_4dct_dec = unzip([treatment_phase_to_4dct(p) for p in treatment_phases])
_, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].scatter(treatment_angles, treatment_phases)
axs[0].set_ylim(0, 360)
axs[1].scatter(treatment_angles, phases_4dct)
axs[1].set_ylim(0, 9)
plt.show()

dataset = 'VALKIM-PP'
inh_series = 'series_0'
exh_series = 'series_5'
set = ds.load(dataset, 'nifti')
pat = set.patient('i:0')
print(pat)

# Create placeholder for projection images and labels.
n_angles = len(treatment_angles)
ct_proj = np.zeros((n_angles, *arc_info[0]['det-size']))
n_label_channels = 2
labels_proj = np.zeros((n_angles, n_label_channels, *arc_info[0]['det-size']))
print(ct_proj.shape)

# Load planned iso.
rtplan = pat.dicom.default_rtplan.dicom
rtplan_info = from_rtplan_dicom(rtplan)
treatment_iso = rtplan_info['isocentre']

# treatment_angles = [i['kv-source-angle'] for i in arc_info]
# treatment_phases = [i['MMPhase0'] for i in arc_info]
# phases_4dct, phases_4dct_dec = unzip([treatment_phase_to_4dct(p) for p in treatment_phases])

# Project for each treatment image!
for i in tqdm(range(len(treatment_angles))):
    angles = treatment_angles[i]
    phase_4dct, phase_4dct_di = phases_4dct[i], phases_4dct_dec[i]
    
    # Load phase data.
    # Inh/exh phases have real GTVs, intermediate phases have corrfield-propagated GTV.
    regions = ['GTV', 'ts_Lung'] if phase_4dct in (0, 5) else ['cf_GTV', 'ts_Lung']
    assert len(regions) == n_label_channels
    phase_series = f'series_{phase_4dct}'
    phase_ct = pat.ct_series(phase_series).data
    phase_affine = pat.ct_series(phase_series).affine
    phase_labels = pat.regions_series(phase_series).data(r=regions, rr=False)

    # Get next phase.
    next_phase_4dct = (phase_4dct + 1) % 10
    next_phase_series = f'series_{next_phase_4dct}'
    next_phase_ct = pat.ct_series(next_phase_series).data
    next_phase_affine = pat.ct_series(next_phase_series).affine
    next_phase_labels = pat.regions_series(next_phase_series).data(r=regions, rr=False)

    # Interpolate the labels.
    # Can we improve the radial interpolation efficiency. E.g. if we pass many "t" values,
    # we get back many interpolated labels. Then we could batch the label production for
    # a single 4DCT phase.
    interp_labels = []
    for la, lb in zip(phase_labels, next_phase_labels):
        interp_label = interpolate_masks_radial_3d(la, lb, phase_4dct_di)
        interp_labels.append(interp_label)
    interp_labels = np.stack(interp_labels, axis=0)

    # Check that all images share projection geometry.
    first_info = arc_info[i]
    sid = first_info['sid']
    sdd = first_info['sdd']
    det_size = first_info['det-size']
    det_spacing = first_info['det-spacing']
    det_offset = first_info['det-offset']
    print(treatment_iso, sid, sdd, det_size, det_spacing, det_offset)
    
    ct_proj_phase, labels_proj_phase = project_ctorch(phase_ct, phase_affine, treatment_iso,
        sid, sdd, det_size, det_spacing, det_offset, angles, labels=interp_labels)
    ct_proj[i] = ct_proj_phase
    labels_proj[i] = labels_proj_phase

save_numpy(ct_proj, f'files:mlm/valkim/gt/projections/{pat.id}/ct_proj_interp.npz')
save_numpy(labels_proj, f'files:mlm/valkim/gt/projections/{pat.id}/labels_proj_interp.npz')