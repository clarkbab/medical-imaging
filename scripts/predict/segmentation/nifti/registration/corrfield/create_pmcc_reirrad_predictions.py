from mymi.predictions.nifti.registration import create_corrfield_predictions

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    # pat_ids='pad_4',
    preprocess_images=True,
)
create_corrfield_predictions(dataset, **kwargs)
