from mymi.predictions.nifti.registration import warp_patients_data

dataset = 'PMCC-REIRRAD-CP'
models = ['velocity-dmp', 'velocity-edmp']
kwargs = dict(
    pat_ids= ['pat_5', 'pat_6'],
)
for m in models:
    warp_patients_data(dataset, m, **kwargs)
