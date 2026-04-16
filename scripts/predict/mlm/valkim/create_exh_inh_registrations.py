import dicomset as ds
from dicomset.nifti.utils import save_registration_transform, save_registration_moved_regions, save_registration_moved_image, load_registration_transform, load_registered_image, load_registered_regions
from tqdm import tqdm

from mymi.predictions.registration import register_corrfield, register_plastimatch, register_unigradicon
from mymi.transforms import resample

methods = ['corrfield', 'plastimatch', 'unigradicon', 'unigradicon-io']

dataset = 'VALKIM-PP'
inh_series = 'series_0'
exh_series = 'series_5'
set = ds.get(dataset, 'nifti')
# pat_ids = ['PAT1', 'PAT2', 'PAT3']
pat_ids = ['PAT3']
regions = ['GTV', 'ts_Lung']

for p in tqdm(pat_ids):
    pat = set.patient(p)
    inh_ct = pat.ct_series(inh_series).data
    inh_affine = pat.ct_series(inh_series).affine
    inh_labels = pat.regions_series(inh_series).data(r=regions)
    exh_ct = pat.ct_series(exh_series).data
    exh_affine = pat.ct_series(exh_series).affine
    exh_labels = pat.regions_series(exh_series).data(r=regions)

    # Register exhale -> inhale using <method>.
    fixed_ct = inh_ct
    moving_ct = exh_ct
    fixed_affine = inh_affine
    moving_affine = exh_affine
    fixed_lung_mask = inh_labels[1]
    moving_lung_mask = exh_labels[1]

    for m in methods:
        # Register <method>.
        if m == 'corrfield':
            transform = register_corrfield(fixed_ct, moving_ct, fixed_affine, moving_affine, fixed_lung_mask=fixed_lung_mask, moving_lung_mask=moving_lung_mask, keep_temp=False, use_io=False)
        elif m == 'plastimatch':
            transform = register_plastimatch(fixed_ct, moving_ct, fixed_affine, moving_affine, keep_temp=False, use_io=False)
        elif m == 'unigradicon':
            transform = register_unigradicon(fixed_ct, moving_ct, fixed_affine, moving_affine, keep_temp=False, use_io=False)
        elif m == 'unigradicon-io':
            transform = register_unigradicon(fixed_ct, moving_ct, fixed_affine, moving_affine, keep_temp=False, use_io=True)
        save_registration_transform(dataset, pat.id, m, transform, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
        moved_ct = resample(exh_ct, affine=exh_affine, output_affine=inh_affine, transform=transform)
        save_registration_moved_image(dataset, pat.id, m, moved_ct, fixed_affine, 'ct', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
        moved_labels = resample(exh_labels, affine=exh_affine, output_affine=inh_affine, transform=transform)
        save_registration_moved_regions(dataset, pat.id, m, moved_labels, fixed_affine, regions, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
