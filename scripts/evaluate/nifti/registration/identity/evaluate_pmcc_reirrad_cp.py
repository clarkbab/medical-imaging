from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'PMCC-REIRRAD-CP'
model = 'identity'
kwargs = dict(
    # splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)
