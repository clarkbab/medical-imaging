from mymi.predictions.nifti import create_deeds_predictions, warp_patients_data

dataset = 'PMCC-REIRRAD-CP'
model = 'deeds'
kwargs = dict(
    preprocess_images=False,
)
# create_deeds_predictions(dataset, **kwargs)
warp_patients_data(dataset, model)
