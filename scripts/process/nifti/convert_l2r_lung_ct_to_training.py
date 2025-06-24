from mymi.processing.nifti import convert_to_registration_training_holdout

dataset = 'L2R-LUNG-CT'
dest_dataset = 'L2R-LUNG-CT-222'
kwargs = dict(
    spacing=(2, 2, 2)
)
convert_to_registration_training_holdout(dataset, dest_dataset, **kwargs)
