from mymi.processing.nifti import convert_to_registration_training_holdout

dataset = 'LUNG250M-4B'
dest_dataset = 'LUNG250M-4B-222'
kwargs = dict(
    spacing=(2, 2, 2)
)
convert_to_registration_training_holdout(dataset, dest_dataset, **kwargs)
