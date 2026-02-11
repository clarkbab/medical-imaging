from mymi.evaluations.nifti import create_registration_evaluations

dataset = 'PMCC-REIRRAD'
model ='deeds'
kwargs = dict(
    # pat_ids='i:0',
    # regions=None,
    # splits='test',
)

create_registration_evaluations(dataset, model, **kwargs)
