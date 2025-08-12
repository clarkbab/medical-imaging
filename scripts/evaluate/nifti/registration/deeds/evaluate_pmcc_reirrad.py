from mymi.evaluations.nifti import create_registration_evaluations

dataset = 'PMCC-REIRRAD'
model ='deeds'
kwargs = dict(
    # pat_ids='idx:0',
    # region_ids=None,
    # splits='test',
)

create_registration_evaluations(dataset, model, **kwargs)
