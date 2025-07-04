from mymi.evaluations.nifti import create_registrations_evaluation

dataset = 'PMCC-REIRRAD-CP'
models = ['velocity-dmp', 'velocity-edmp']
kwargs = dict(
    # splits='test',
)
for m in models:
    create_registrations_evaluation(dataset, m, **kwargs)
