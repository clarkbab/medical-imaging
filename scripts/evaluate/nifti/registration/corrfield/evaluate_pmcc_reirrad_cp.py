from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'PMCC-REIRRAD-CP'
model ='corrfield'
kwargs = dict(
    exclude_pat_ids=['pat_0', 'pat_7', 'pat_9'],
    # splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)
