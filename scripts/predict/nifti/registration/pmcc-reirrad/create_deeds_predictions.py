from mymi.predictions.nifti import create_deeds_predictions

dataset = 'PMCC-REIRRAD-CP'
model = 'deeds'
kwargs = dict(
    pat_ids=['PMCC_ReIrrad_L03', 'PMCC_ReIrrad_L08', 'PMCC_ReIrrad_L14'],
    preprocess_images=False,
)
create_deeds_predictions(dataset, **kwargs)
