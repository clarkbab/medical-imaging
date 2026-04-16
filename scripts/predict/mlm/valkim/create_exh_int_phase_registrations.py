import dicomset as ds
from dicomset.nifti.utils import create_registration_transform, create_registered_regions, create_registered_image
from tqdm import tqdm

from mymi.predictions.registration import register_corrfield, register_linear, register_plastimatch, register_unigradicon
from mymi.transforms import resample

# methods = ['corrfield', 'plastimatch', 'unigradicon', 'unigradicon-io']
methods = ['corrfield', 'linear']
# methods = ['unigradicon', 'unigradicon-io']

dataset = 'VALKIM-PP'
other_series = ['series_1', 'series_2', 'series_3', 'series_4']
# other_series = ['series_9', 'series_8', 'series_7', 'series_6']
t_vals = [0.2, 0.4, 0.6, 0.8]
# other_series = ['series_1']
inh_series = 'series_0'
exh_series = 'series_5'
set = ds.get(dataset, 'nifti')
# pat_ids = ['PAT1', 'PAT2', 'PAT3']
pat_ids = ['PAT1']
regions = ['GTV', 'ts_Lung']
other_regions = ['ts_Lung']    # No GTV label for intermediate phases - yet...

for p in tqdm(pat_ids):
    pat = set.patient(p)
    inh_ct = pat.ct_series(inh_series).data
    inh_affine = pat.ct_series(inh_series).affine
    inh_labels = pat.regions_series(inh_series).data(r=regions)
    exh_ct = pat.ct_series(exh_series).data
    exh_affine = pat.ct_series(exh_series).affine
    exh_labels = pat.regions_series(exh_series).data(r=regions)
    for s, t in tqdm(zip(other_series, t_vals), leave=False): 
        other_ct = pat.ct_series(s).data
        other_affine = pat.ct_series(s).affine
        other_labels = pat.regions_series(s).data(r=other_regions)

        # Register exhale -> inhale using <method>.
        fixed_ct = other_ct
        moving_ct = exh_ct
        fixed_affine = other_affine
        moving_affine = exh_affine

        for m in methods:
            # Register <method>.
            if m == 'corrfield':
                assert other_regions[0] == 'ts_Lung', "Currently hardcoded to use the first region as the lung mask for corrfield registration."
                assert regions[1] == 'ts_Lung', "Currently hardcoded to use the second region as the lung mask for corrfield registration."                
                fixed_lung_mask = other_labels[0]
                moving_lung_mask = exh_labels[1]
                transform = register_corrfield(fixed_ct, moving_ct, fixed_affine, moving_affine, fixed_lung_mask=fixed_lung_mask, moving_lung_mask=moving_lung_mask)
            elif m == 'linear':
                assert regions[0] == 'GTV', "Currently hardcoded to use the first region as the GTV for linear registration." 
                inh_gtv = inh_labels[0]
                exh_gtv = exh_labels[0] 
                moved_gtv = register_linear(inh_gtv, exh_gtv, fixed_affine, moving_affine, t=t)   
            elif m == 'plastimatch':
                transform = register_plastimatch(fixed_ct, moving_ct, fixed_affine, moving_affine)
            elif m == 'unigradicon':
                transform = register_unigradicon(fixed_ct, moving_ct, fixed_affine, moving_affine)
            elif m == 'unigradicon-io':
                transform = register_unigradicon(fixed_ct, moving_ct, fixed_affine, moving_affine, use_io=True)

            # Save results.
            if m != 'linear':
                create_registration_transform(dataset, pat.id, m, transform, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)
                moved_ct = resample(exh_ct, affine=exh_affine, output_affine=other_affine, transform=transform)
                create_registered_image(dataset, pat.id, m, moved_ct, fixed_affine, 'ct', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)
                moved_labels = resample(exh_labels, affine=exh_affine, output_affine=other_affine, transform=transform)
                create_registered_regions(dataset, pat.id, m, moved_labels, fixed_affine, regions, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)
            else:
                create_registered_regions(dataset, pat.id, m, moved_gtv, fixed_affine, 'GTV', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=s, moving_series_id=exh_series)
