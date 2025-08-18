from mymi.predictions.nifti.registration import create_corrfield_predictions, warp_patients_data

dataset = 'PMCC-REIRRAD-CP'
model = 'corrfield'
kwargs = dict(
    # pat_ids='pad_4',
    preprocess_images=False,
)
create_corrfield_predictions(dataset, **kwargs)
# warp_patients_data(dataset, model)
