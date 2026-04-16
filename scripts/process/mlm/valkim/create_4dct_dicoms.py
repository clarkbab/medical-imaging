import dicomset as ds
from dicomset.dicom.utils import save_dicom, to_ct_dicom, to_rtstruct_dicom
from dicomset.nifti.utils import load_registered_regions
import numpy as np
from tqdm import tqdm

dataset = 'VALKIM-PP'
pat_ids = ['PAT1', 'PAT2', 'PAT3']
inh_series = 'series_0'
exh_series = 'series_5'
set = ds.get(dataset, 'nifti')

for p in tqdm(pat_ids):
    # Load inhale/exhale data.
    pat = set.patient(p)
    inh_ct = pat.ct_series(inh_series).data
    inh_affine = pat.ct_series(inh_series).affine
    inh_gtv = pat.regions_series(inh_series).data(r='GTV')[0]
    exh_ct = pat.ct_series(exh_series).data
    exh_affine = pat.ct_series(exh_series).affine
    exh_gtv = pat.regions_series(exh_series).data(r='GTV')[0]

    # Load intermediate phases.
    # int_phases = ['series_1', 'series_2', 'series_3', 'series_4',
    #     'series_6', 'series_7', 'series_8', 'series_9']
    int_phases = ['series_1', 'series_2', 'series_3', 'series_4']
    int_cts = [pat.ct_series(s).data for s in int_phases]
    int_affines = [pat.ct_series(s).affine for s in int_phases]

    # Add other phases.
    moved_gtvs = []
    linear_moved_gtvs = []
    for s in int_phases:
        # Load corrfield registered GTV.
        moved_gtv, _ = load_registered_regions(dataset, pat.id, 'corrfield', 'GTV', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)
        moved_gtvs.append(moved_gtv)

        # Load linear registered GTV.
        moved_gtv, _ = load_registered_regions(dataset, pat.id, 'linear', 'GTV', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)
        linear_moved_gtvs.append(moved_gtv)

    # Save inhale dicom files.
    dirpath = f'files:mlm/valkim/gt/{pat.id}/phase_0/ct'
    ct_dicoms = to_ct_dicom(inh_ct, inh_affine, patient_id=pat.id, series_desc='Phase 0 (inhale)', series_number=0)
    save_dicom(ct_dicoms, dirpath)

    study_uid = ct_dicoms[0].StudyInstanceUID
    filepath = f'files:mlm/valkim/gt/{pat.id}/phase_0/rtstruct.dcm'
    rtstruct = to_rtstruct_dicom(inh_gtv, 'GTV', ct_dicoms, series_desc='Phase 0 (inhale)', series_number=0)
    save_dicom(rtstruct, filepath)

    # Save exhale dicom files.
    dirpath = f'files:mlm/valkim/gt/{pat.id}/phase_5/ct'
    ct_dicoms = to_ct_dicom(exh_ct, exh_affine, patient_id=pat.id, series_desc='Phase 5 (exhale)', study_uid=study_uid, series_number=5)
    save_dicom(ct_dicoms, dirpath)

    filepath = f'files:mlm/valkim/gt/{pat.id}/phase_5/rtstruct.dcm'
    rtstruct = to_rtstruct_dicom(exh_gtv, 'GTV', ct_dicoms, series_desc='Phase 5 (exhale)', series_number=5)
    save_dicom(rtstruct, filepath)

    # Save intermediate phases.
    for i, s in enumerate(int_phases):
        phase = int(s.split('_')[1])
        dirpath = f'files:mlm/valkim/gt/{pat.id}/phase_{phase}/ct'
        ct_dicoms = to_ct_dicom(int_cts[i], int_affines[i], patient_id=pat.id, series_desc=f'Phase {phase}', study_uid=study_uid, series_number=phase)
        save_dicom(ct_dicoms, dirpath)
        
        filepath = f'files:mlm/valkim/gt/{pat.id}/phase_{phase}/rtstruct.dcm'
        moved_labels = np.stack([moved_gtvs[i], linear_moved_gtvs[i], exh_gtv], axis=0)
        region_ids = ['GTV (corrfield)', 'GTV (linear)', 'GTV (exhale)']
        rtstruct = to_rtstruct_dicom(moved_labels, region_ids, ct_dicoms, series_desc=f'Phase {phase}', series_number=phase)
        save_dicom(rtstruct, filepath)
