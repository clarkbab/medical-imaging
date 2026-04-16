
from dicomset.nifti.utils import save_registration_transform, save_registration_moved_regions, save_registration_moved_image, load_registration_transform, load_registered_image, load_registered_regions
from mymi.predictions.registration import register_corrfield, register_plastimatch, register_unigradicon
from mymi.transforms import resample

# Register exhale -> inhale using unigradicon.
fixed_ct = inh_ct
moving_ct = exh_ct
fixed_affine = inh_affine
moving_affine = exh_affine

# Register unigradicon.
transform = register_unigradicon(fixed_ct, moving_ct, fixed_affine, moving_affine, keep_temp=False, use_io=True)
save_registration_transform(dataset, pat.id, 'unigradicon-io', transform, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
moved_ct = resample(exh_ct, affine=exh_affine, output_affine=inh_affine, transform=transform)
save_registration_moved_image(dataset, pat.id, 'unigradicon-io', moved_ct, fixed_affine, 'ct', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
moved_labels = resample(exh_labels, affine=exh_affine, output_affine=inh_affine, transform=transform)
save_registration_moved_regions(dataset, pat.id, 'unigradicon-io', moved_labels, fixed_affine, regions, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)

transform = load_registration_transform(dataset, pat.id, 'unigradicon-io', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
moved_ct, _ = load_registered_image(dataset, pat.id, 'unigradicon-io', 'ct', fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)
moved_labels, _ = load_registered_regions(dataset, pat.id, 'unigradicon-io', regions, fixed_study_id='study_0', moving_study_id='study_0', fixed_series_id=inh_series, moving_series_id=exh_series)