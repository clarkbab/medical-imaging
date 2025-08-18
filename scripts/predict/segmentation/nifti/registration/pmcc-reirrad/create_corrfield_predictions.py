from mymi.predictions.nifti.registration import create_corrfield_predictions

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    # pat_ids=['PMCC_ReIrrad_L03', 'PMCC_ReIrrad_L08', 'PMCC_ReIrrad_L14'],
    pat_ids='PMCC_ReIrrad_L01',
    preprocess_images=False,
)
create_corrfield_predictions(dataset, **kwargs)
