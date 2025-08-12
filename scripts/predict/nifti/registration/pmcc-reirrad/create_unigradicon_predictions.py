from mymi.predictions.nifti import create_unigradicon_predictions

dataset = 'PMCC-REIRRAD-CP'
model = 'unigradicon'
kwargs = dict(
    pat_ids=['PMCC_ReIrrad_L03', 'PMCC_ReIrrad_L08', 'PMCC_ReIrrad_L14'],
)
create_unigradicon_predictions(dataset, model, **kwargs)
