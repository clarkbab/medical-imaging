from mymi.predictions.nifti import create_registrations

dataset = "PMCC-REIRRAD-CP"
project = "IMREG"
model = 'identity'
model_spacing = None
kwargs = dict(
    pat_ids=['PMCC_ReIrrad_L03', 'PMCC_ReIrrad_L08', 'PMCC_ReIrrad_L14'],
)

create_registrations(dataset, project, model, model_spacing, **kwargs)
